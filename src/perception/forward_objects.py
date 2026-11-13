"""Forward-object warning (M2): YOLO COCO detection + ByteTrack + image-plane zone + TTC proxy.

Single camera, no metric distance. The TTC value is a relative approach indicator
(box height growth), NOT a time-to-collision. Wording in speech must not claim a collision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# COCO class ids used by models/lane/candidates/coco_yolo11n_20261011.pt
COCO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
COCO_VI = {'person': 'người đi bộ', 'bicycle': 'xe đạp', 'car': 'ô tô', 'motorcycle': 'xe máy',
           'bus': 'xe buýt', 'truck': 'xe tải'}


@dataclass
class ForwardWarnConfig:
    enabled: bool = False
    weights_path: str = 'models/lane/candidates/coco_yolo11n_20261011.pt'
    imgsz: int = 640
    conf: float = 0.35
    bottom_width_ratio: float = 0.50   # trapezoid base width, fraction of frame width
    top_width_ratio: float = 0.16      # trapezoid top width
    top_y_ratio: float = 0.55          # top edge of the zone, fraction of H measured from the top
    chu_y_bottom_ratio: float = 0.75   # bottom_y / H for level chu_y
    chu_y_persist: int = 3             # consecutive samples
    nguy_co_ttc_s: float = 2.0
    nguy_co_persist: int = 2
    ttc_k: int = 3                     # samples between size ratios (~0.5 s at 6 Hz)
    ttc_min_scale: float = 1.03
    cooldown_track_s: float = 6.0
    cooldown_global_s: float = 3.0

    @classmethod
    def from_dict(cls, d: dict | None) -> 'ForwardWarnConfig':
        d = dict(d or {})
        zone = d.pop('zone', {}) or {}
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.bottom_width_ratio = zone.get('bottom_width_ratio', cfg.bottom_width_ratio)
        cfg.top_width_ratio = zone.get('top_width_ratio', cfg.top_width_ratio)
        cfg.top_y_ratio = zone.get('top_y_ratio', cfg.top_y_ratio)
        return cfg


def in_zone(point_xy, width, height, cfg: ForwardWarnConfig) -> bool:
    """True when the point lies inside the trapezoid: base at the bottom edge, top at top_y_ratio."""
    x, y = point_xy
    top_y = cfg.top_y_ratio * height
    if y < top_y or y > height:
        return False
    # Linear half-width from top (top_width/2) to bottom (bottom_width/2), centred on width/2.
    t = (y - top_y) / (height - top_y)
    half = (cfg.top_width_ratio + t * (cfg.bottom_width_ratio - cfg.top_width_ratio)) * width / 2
    return abs(x - width / 2) <= half


def ttc_proxy(heights: list[float], k: int, dt_s: float, min_scale: float) -> float | None:
    """Relative approach indicator from box-height growth: s = h_t / h_{t-k}; TTC ~ k*dt / (s-1)."""
    if len(heights) <= k or heights[-1 - k] <= 0:
        return None
    s = heights[-1] / heights[-1 - k]
    if s <= min_scale:
        return None
    return k * dt_s / (s - 1.0)


@dataclass
class _Track:
    heights: deque = field(default_factory=lambda: deque(maxlen=16))
    ema: float | None = None
    chu_y_run: int = 0
    nguy_run: int = 0
    last_alert_t: float = -1e9
    last_level: str | None = None


class ForwardObjectWarner:
    """Stateful per-job analyser. Feed it the detections of each analysis sample."""

    def __init__(self, cfg: ForwardWarnConfig, analysis_hz: float):
        self.cfg = cfg
        self.dt = 1.0 / analysis_hz
        self.tracks: dict[int, _Track] = {}
        self.last_global_t = -1e9

    def reset(self):
        self.tracks.clear()
        self.last_global_t = -1e9

    def update(self, t_s: float, width: int, height: int, detections) -> list[dict]:
        """detections: iterable of dicts {track_id, cls, bbox_xyxy, conf}. Returns alerts for this sample."""
        if not self.cfg.enabled:
            return []
        alerts = []
        seen = set()
        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                continue
            seen.add(tid)
            x1, y1, x2, y2 = det['bbox_xyxy']
            h = max(0.0, float(y2 - y1))
            tr = self.tracks.setdefault(tid, _Track())
            tr.ema = h if tr.ema is None else 0.5 * tr.ema + 0.5 * h   # height smoothing
            tr.heights.append(tr.ema)
            bottom_mid = ((x1 + x2) / 2.0, float(y2))
            inside = in_zone(bottom_mid, width, height, self.cfg)
            near = (y2 / height) >= self.cfg.chu_y_bottom_ratio
            ttc = ttc_proxy(list(tr.heights), self.cfg.ttc_k, self.dt, self.cfg.ttc_min_scale)
            tr.chu_y_run = tr.chu_y_run + 1 if (inside and near) else 0
            tr.nguy_run = tr.nguy_run + 1 if (inside and ttc is not None and ttc < self.cfg.nguy_co_ttc_s) else 0
            level = None
            if tr.nguy_run >= self.cfg.nguy_co_persist:
                level = 'nguy_co'
            elif tr.chu_y_run >= self.cfg.chu_y_persist:
                level = 'chu_y'
            if level is None:
                continue
            # Escalation to nguy_co is never held back by cooldown (safety-relevant upgrade).
            escalated = level == 'nguy_co' and tr.last_level != 'nguy_co'
            if not escalated:
                if t_s - tr.last_alert_t < self.cfg.cooldown_track_s:
                    continue
                if t_s - self.last_global_t < self.cfg.cooldown_global_s:
                    continue
            tr.last_alert_t = t_s
            tr.last_level = level
            self.last_global_t = t_s
            cls_name = COCO_CLASSES.get(det.get('cls'), str(det.get('cls')))
            alerts.append({'t': round(t_s, 3), 'track_id': tid, 'cls': cls_name,
                           'label_vi': COCO_VI.get(cls_name, cls_name), 'level': level,
                           'ttc_proxy_s': None if ttc is None else round(ttc, 3),
                           'bbox_xyxy': [float(v) for v in (x1, y1, x2, y2)]})
        # Tracks not seen this sample lose their run counters (no stale hysteresis).
        for tid, tr in self.tracks.items():
            if tid not in seen:
                tr.chu_y_run = 0
                tr.nguy_run = 0
        return alerts


def speech_text(level: str, cls_name: str) -> str:
    """Fixed cue sentences. Never use collision words."""
    vi = COCO_VI.get(cls_name, 'phương tiện')
    if level == 'nguy_co':
        return f'Cẩn thận, {vi} đang tiến gần'
    return f'Chú ý, {vi} phía trước'


PRIORITY = {'nguy_co': 3, 'chu_y': 1}


def pick_speech(cues: list[dict], sign_cues_present: bool) -> dict | None:
    """Priority: nguy_co > traffic sign cue > chu_y. Only one cue per sample is spoken."""
    if not cues:
        return None
    best = max(cues, key=lambda c: PRIORITY.get(c['level'], 0))
    if best['level'] == 'nguy_co':
        return best
    return None if sign_cues_present else best


class CocoTracker:
    """YOLO COCO detector + ByteTrack. One instance per video job (call reset_job between jobs)."""

    def __init__(self, cfg: ForwardWarnConfig, device: str = 'auto'):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.weights_path)
        self.device = device
        self.classes = list(COCO_CLASSES)

    def reset_job(self):
        predictor = getattr(self.model, 'predictor', None)
        trackers = getattr(predictor, 'trackers', None) if predictor is not None else None
        if trackers:
            for tr in trackers:
                tr.reset()

    def detect(self, frame):
        results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', imgsz=self.cfg.imgsz,
                                   conf=self.cfg.conf, classes=self.classes, verbose=False,
                                   device=None if self.device == 'auto' else self.device)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for box, tid in zip(r.boxes, ids):
                out.append({'track_id': None if tid is None else int(tid),
                            'cls': int(box.cls.item()), 'conf': float(box.conf.item()),
                            'bbox_xyxy': box.xyxy[0].tolist()})
        return out


def forward_cues(alerts: list[dict], sign_present: bool, t_s: float) -> list[dict]:
    """Speech cues for one sample. nguy_co always spoken; chu_y only if no traffic-sign cue this sample."""
    cues = [{'t_seconds': round(t_s, 3), 'codes': [], 'text': speech_text('nguy_co', a['cls']),
             'kind': 'forward_nguy_co'} for a in alerts if a['level'] == 'nguy_co']
    chu = [a for a in alerts if a['level'] == 'chu_y']
    if chu and not sign_present:
        cues.append({'t_seconds': round(t_s, 3), 'codes': [], 'text': speech_text('chu_y', chu[0]['cls']),
                     'kind': 'forward_chu_y'})
    return cues


def merge_cues(sign_cues: list[dict], forward: list[dict], busy_s: float = 2.0) -> list[dict]:
    """Order cues on one timeline. Forward nguy_co goes first; sign cues that would overlap it are
    moved to after it (not dropped), so cues never talk over each other."""
    out = [dict(c) for c in sign_cues]
    nguy = sorted((c for c in forward if c.get('kind') == 'forward_nguy_co'), key=lambda c: c['t_seconds'])
    for f in nguy:
        # Pushed sign cues keep their order and never share a start time.
        cursor = f['t_seconds'] + busy_s
        for c in sorted(out, key=lambda c: c['t_seconds']):
            if f['t_seconds'] <= c['t_seconds'] < cursor:
                c['t_seconds'] = round(cursor, 3)
                cursor = c['t_seconds'] + 0.5
    out += [dict(c) for c in forward]
    return sorted(out, key=lambda c: c['t_seconds'])


def draw_forward(img, alerts: list[dict], cfg: ForwardWarnConfig, font):
    """Overlay: zone trapezoid (faint), object boxes orange (chu_y) / red (nguy_co) with Vietnamese labels."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    zone = np.array([[int(w * (0.5 - cfg.bottom_width_ratio / 2)), h - 1],
                     [int(w * (0.5 - cfg.top_width_ratio / 2)), int(cfg.top_y_ratio * h)],
                     [int(w * (0.5 + cfg.top_width_ratio / 2)), int(cfg.top_y_ratio * h)],
                     [int(w * (0.5 + cfg.bottom_width_ratio / 2)), h - 1]], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [zone], (255, 255, 0))
    img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    if not alerts:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(pil)
    for a in alerts:
        x1, y1, x2, y2 = [round(v) for v in a['bbox_xyxy']]
        color = (255, 0, 0) if a['level'] == 'nguy_co' else (255, 140, 0)
        painter.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = ('NGUY CƠ: ' if a['level'] == 'nguy_co' else 'CHÚ Ý: ') + a['label_vi']
        painter.text((x1, max(0, y1 - 22)), label, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
