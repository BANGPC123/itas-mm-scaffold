"""A4: recognizer backbone comparison on the recognizer-v2 crops (2026-11-12).

Same data, classes, sampler, schedule and seed as scripts/train_recognizer_v2_20261028.py; only the
backbone changes. Every backbone starts from torchvision ImageNet weights with a fresh classifier
(the 2026-10-09 checkpoint is NOT used as initialisation). VAL selects the epoch and the acceptance
threshold; TEST is read once, after the selection, for reporting.

--reference evaluates the deployed checkpoint models/sign_recognizer/retrained_20261028.pt on TEST with
the same code (no training), so every row of the table comes from one script.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as tv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.perception.recognizer_v2_crops import build_cache, exact_crop, load_cache, save_cache  # noqa: E402
from src.perception.sign_recognizer import crop_tensor  # noqa: E402
from train_recognizer_v2_20261028 import (TrainCrops, calibrate, evaluate, predict)  # noqa: E402

V2 = ROOT / "output/kiet_wp_20261028/recognizer_v2"
LOC = ROOT / "data/processed/canonical/itas_mm_v1/localization"
DEPLOYED = ROOT / "models/sign_recognizer/retrained_20261028.pt"
OLD = ROOT / "models/sign_recognizer/retrained_20261009.pt"
OUT_ROOT = ROOT / "output/bench_20261112/A4"
MIN_SIDE = 16
SEED = 20261028
GROUPS = [("<16", 0, 16), ("16-32", 16, 32), ("32-64", 32, 64), (">=64", 64, 1e9)]


def build_model(backbone: str, num_classes: int) -> torch.nn.Module:
    if backbone == "mobilenet_v3_small":
        model = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        model = tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "resnet18":
        model = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(backbone)
    return model


def read_split(split: str, classes: list[str]):
    """Rows and cached crops for one split, without writing to the v2 folder.

    TEST follows evaluate_recognizer_v2_20261028.py: every TEST box, cache crop_cache_test_all.pkl.
    """
    boxes = pd.read_csv(V2 / "boxes.csv", keep_default_na=False)
    background = pd.read_csv(V2 / "background_boxes.csv", keep_default_na=False)
    if split == "TEST":
        rows = boxes[boxes.split == "TEST"].reset_index(drop=True)
        cache = V2 / "crop_cache_test_all.pkl"
        entries = load_cache(cache) if cache.is_file() else None
        if entries is None or len(entries) != len(rows):
            raise RuntimeError(f"Missing or stale crop cache {cache.name}; refusing to write into the v2 folder")
        keep = [i for i, e in enumerate(entries) if e is not None]
        return rows.iloc[keep].reset_index(drop=True), [entries[i] for i in keep]
    signs = boxes[(boxes.split == split) & (boxes.short_side_px >= MIN_SIDE) & boxes.code.isin(classes)]
    bg = background[background.split == split]
    cols = ["image_path", "bbox_xyxy", "short_side_px", "code", "source"]
    rows = pd.concat([signs[cols], bg[cols]], ignore_index=True)
    cache = V2 / f"crop_cache_{split.lower()}.pkl"
    entries = load_cache(cache) if cache.is_file() else None
    if entries is None or len(entries) != len(rows):
        raise RuntimeError(f"Missing or stale crop cache {cache.name}; refusing to write into the v2 folder")
    keep = [i for i, e in enumerate(entries) if e is not None]
    return rows.iloc[keep].reset_index(drop=True), [entries[i] for i in keep]


def metrics_block(true: np.ndarray, pred: np.ndarray) -> dict:
    labels = sorted(set(true.tolist()))
    return dict(n=int(len(true)), classes=len(labels),
                accuracy=float((true == pred).mean()) if len(true) else None,
                macro_f1=float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)) if len(true) else None)


def summarize_test(pred_names: np.ndarray, rows: pd.DataFrame, old30: list[str], main_classes: list[str]) -> dict:
    """Same tables as evaluate_recognizer_v2_20261028.py: subset old30 (codes of the 2026-10-09 model) and all TEST."""
    true = rows.code.to_numpy()
    sizes = rows.short_side_px.to_numpy()
    out = {}
    for subset_name, mask in (("old30", np.isin(true, old30)), ("all_test", np.ones(len(true), bool))):
        table = {}
        for name, lo, hi in GROUPS + [(">=32", 32, 1e9), ("all", 0, 1e9)]:
            sel = mask & (sizes >= lo) & (sizes < hi)
            table[name] = metrics_block(true[sel], pred_names[sel])
        out[subset_name] = table
    out["main_classes_in_test"] = int(np.isin(true, main_classes).sum())
    return out


def run_reference(out: Path) -> dict:
    report = json.loads((V2 / "class_list.json").read_text(encoding="utf-8"))
    classes, main_classes = report["classes"], report["main"]
    rows, entries = read_split("TEST", classes)
    ckpt = torch.load(DEPLOYED, map_location="cpu", weights_only=True)
    model = tv.mobilenet_v3_small(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    logits = predict(model, entries, device)
    pred = logits.softmax(1).argmax(1).numpy()
    pred_names = np.asarray([ckpt["classes"][i] for i in pred])
    old30 = [c for c in torch.load(OLD, map_location="cpu", weights_only=True)["classes"] if c != "unknown"]
    result = {"model": "retrained_20261028.pt (deployed, not retrained)", "params": int(sum(p.numel() for p in model.parameters())),
              "threshold_deployed": float(ckpt.get("confidence_threshold", 0.87)),
              "test": summarize_test(pred_names, rows, old30, main_classes),
              "note": "TEST labels AI-assisted; no selection on TEST; raw argmax as in the 2026-10-28 evaluation"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_train(backbone: str, out: Path, epochs: int, batch: int, epoch_size: int, lr: float) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    report = json.loads((V2 / "class_list.json").read_text(encoding="utf-8"))
    classes, main_classes = report["classes"], report["main"]
    bg = classes.index("unknown")
    started = time.time()
    train_rows, train_entries = read_split("TRAIN", classes)
    val_rows, val_entries = read_split("VAL", classes)
    ytr = train_rows.code.map(classes.index).to_numpy()
    yva = val_rows.code.map(classes.index).to_numpy()
    sizes = val_rows.short_side_px.to_numpy()
    counts = np.bincount(ytr, minlength=len(classes))
    weights = (counts[ytr].astype(float)) ** (-0.7)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=epoch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(TrainCrops(train_entries, ytr, SEED), batch_size=batch, sampler=sampler,
                        num_workers=0, drop_last=True)
    model = build_model(backbone, len(classes)).to(device)
    params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (epoch_size // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.08,
                                                anneal_strategy="cos")
    lossfn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    out.mkdir(parents=True, exist_ok=True)
    history, best, best_logits = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        losses, t0 = [], time.time()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        logits = predict(model, val_entries, device)
        metrics = evaluate(logits, yva, classes, main_classes, sizes, bg)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.time() - t0, **metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (metrics["macro_f1_main"], metrics["accuracy_known"])
        if best is None or key > best[0]:
            best = (key, epoch)
            best_logits = logits
            torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            classes=classes, image_size=128, backbone=backbone, background_class=True,
                            main_classes=main_classes, selected_epoch=epoch, seed=SEED),
                       out / "candidate.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    chosen, curve, met = calibrate(best_logits, yva, sizes, bg)
    ckpt = torch.load(out / "candidate.pt", map_location="cpu", weights_only=True)
    ckpt["confidence_threshold"] = float(chosen["threshold"])
    torch.save(ckpt, out / "candidate.pt")
    (out / "calibration.json").write_text(json.dumps(dict(selected=chosen, criterion_met=met, split="VAL",
        criterion="same rule as the 2026-10-28 recognizer (VAL only)", test_used=False), indent=1), encoding="utf-8")
    # One TEST read, after every choice above is fixed.
    test_rows, test_entries = read_split("TEST", classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_logits = predict(model, test_entries, device)
    pred_idx = test_logits.softmax(1).argmax(1).numpy()
    old30 = [c for c in torch.load(OLD, map_location="cpu", weights_only=True)["classes"] if c != "unknown"]
    test = summarize_test(np.asarray([classes[i] for i in pred_idx]), test_rows, old30, main_classes)
    val_best = {"epoch": best[1], "macro_f1_main": best[0][0], "accuracy_known": best[0][1]}
    result = {"backbone": backbone, "params": params, "init": "torchvision ImageNet1K (new classifier)",
              "command": " ".join(sys.argv), "seed": SEED, "epochs": epochs, "batch": batch,
              "epoch_size": epoch_size, "lr": lr, "selected_epoch_val": val_best,
              "threshold_val_calibrated": float(chosen["threshold"]), "threshold_criterion_met": bool(met),
              "val_crops": int(len(val_rows)), "train_crops": int(len(train_rows)),
              "test": test, "test_used_once_after_selection": True,
              "seconds_total_shared_gpu": time.time() - started, "device": device,
              "label_note": "AI-assisted labels, not human ground truth"}
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18"])
    parser.add_argument("--reference", action="store_true", help="evaluate the deployed checkpoint on TEST only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch-size", type=int, default=14000)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    if args.reference:
        out = args.out or OUT_ROOT / "reference_retrained_20261028"
        run_reference(out)
    else:
        if not args.backbone:
            parser.error("--backbone or --reference is required")
        out = args.out or OUT_ROOT / args.backbone
        run_train(args.backbone, out, args.epochs, args.batch, args.epoch_size, args.lr)


if __name__ == "__main__":
    main()
