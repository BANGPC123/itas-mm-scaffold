from pathlib import Path
import math
import cv2
import numpy as np


class SignRecognizer:
    def __init__(self, config):
        self.config = config.get('sign_recognizer', {})
        self.model = None
        self.reference_gate=None
        self.threshold = self.config.get('confidence_threshold', .7)
        path = Path(self.config.get('weights_path', 'models/sign_recognizer/best.pt'))
        if path.is_file():
            import torch
            from torchvision.models import mobilenet_v3_small
            requested=self.config.get('device','auto')
            self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            self.classes = checkpoint['classes']
            self.size = checkpoint.get('image_size', 128)
            self.threshold = checkpoint.get('confidence_threshold', self.threshold)
            self.model = mobilenet_v3_small(num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device).eval()
            reference=self.config.get('reference_gate_path')
            if reference and Path(reference).is_file():
                from src.perception.sign_open_set import FrozenReferenceGate
                self.reference_gate=FrozenReferenceGate(reference,self.device,
                    margin=self.config.get('reference_gate_margin',.015))

    def recognize(self, image, signs, *, min_short_side=None):
        if self.model is None:
            for sign in signs:
                sign.label = 'unknown'
            return signs
        import torch
        eligible, tensors = [], []
        for sign in signs:
            x1,y1,x2,y2 = sign.bbox_xyxy
            h,w = image.shape[:2]
            crop = image[max(0,math.floor(y1)):min(h,math.ceil(y2)),
                         max(0,math.floor(x1)):min(w,math.ceil(x2))]
            required_size=self.config.get('min_short_side',32) if min_short_side is None else min_short_side
            if crop.size == 0 or min(crop.shape[:2]) < required_size:
                sign.label = 'unknown'
                sign.recognition_source = 'crop_too_small'
                continue
            eligible.append(sign)
            tensors.append(crop_tensor(crop, self.size))
        if tensors:
            gate=getattr(self,'reference_gate',None)
            with torch.inference_mode():
                batch=torch.stack(tensors).to(self.device)
                if gate is not None:
                    features=self.model.avgpool(self.model.features(batch)).flatten(1)
                    embedding=self.model.classifier[:-1](features)
                    probabilities=self.model.classifier[-1](embedding).softmax(1)
                else:probabilities = self.model(batch).softmax(1)
                confidences, indices = probabilities.max(1)
                predictions = zip(confidences.cpu().tolist(), indices.cpu().tolist())
                veto=gate.verdict(embedding,[self.classes[i] for i in indices.cpu().tolist()]) if gate is not None else [None]*len(eligible)
            for sign, (confidence, index), rejected in zip(eligible, predictions,veto):
                sign.recognition_confidence = confidence
                sign.label = self.classes[index] if confidence >= self.threshold else 'unknown'
                sign.recognition_source = 'single_frame' if sign.label!='unknown' else 'low_confidence'
                if self.classes[index] == 'unknown' and confidence >= self.threshold:
                    sign.recognition_source = ('out_of_vocabulary_candidate'
                        if self.keeps_confident_detection(sign) else 'background_rejected')
                elif rejected:
                    sign.label='unknown';sign.recognition_source=rejected
                if sign.label=='I.409':
                    from src.perception.traffic_symbols import blue_symbol_verdict
                    x1,y1,x2,y2=sign.bbox_xyxy;h,w=image.shape[:2]
                    crop=image[max(0,math.floor(y1)):min(h,math.ceil(y2)),
                               max(0,math.floor(x1)):min(w,math.ceil(x2))]
                    reason=blue_symbol_verdict(crop,sign.label)
                    if reason:sign.label='unknown';sign.recognition_source=reason
        return signs

    def keeps_confident_detection(self, sign):
        """Keep a confident detector box whose code is outside the class list.

        The recognizer has 30 codes plus a background class, so real signs such
        as I.434a (bus stop) are scored as background and would be erased.
        When enabled, a box with detector confidence and short side at or above
        the configured limits stays visible as an unrecognised sign instead.
        Disabled unless keep_background_detector_confidence is configured.
        """
        limit = self.config.get('keep_background_detector_confidence')
        if limit is None:
            return False
        x1, y1, x2, y2 = sign.bbox_xyxy
        side = min(x2 - x1, y2 - y1)
        return sign.confidence >= limit and side >= self.config.get('keep_background_min_side', 16)


def crop_tensor(bgr, size=128):
    import torch
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]; ratio=size/max(h,w)
    resized=cv2.resize(rgb,(max(1,round(w*ratio)),max(1,round(h*ratio))))
    canvas=np.full((size,size,3),127,dtype=np.uint8)
    y=(size-resized.shape[0])//2;x=(size-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    t=torch.from_numpy(canvas.copy()).permute(2,0,1).float()/255
    return (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
