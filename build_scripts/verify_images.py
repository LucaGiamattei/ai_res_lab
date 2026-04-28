"""Run MobileNetV2 on each sample image and report top-5 predictions.

If an image's expected class is not within top-5, the source needs to be replaced.
"""
from __future__ import annotations
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "challenge_2_adversarial" / "data" / "images"
CLASSES_FILE = ROOT / "challenge_2_adversarial" / "data" / "imagenet_classes.txt"

EXPECTED = {
    "panda.jpg":            (388, "giant panda"),
    "school_bus.jpg":       (779, "school bus"),
    "golden_retriever.jpg": (207, "golden retriever"),
    "traffic_light.jpg":    (920, "traffic light"),
    "espresso.jpg":         (967, "espresso"),
}


def main() -> int:
    classes = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
    assert len(classes) == 1000, len(classes)

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights).eval()
    preprocess = weights.transforms()

    n_top1 = n_top5 = 0
    for fname, (eid, ename) in EXPECTED.items():
        img = Image.open(IMG_DIR / fname).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, 5)
        top5_ids = top5.indices.tolist()
        top5_names = [classes[i] for i in top5_ids]
        top5_probs = top5.values.tolist()
        in_top1 = top5_ids[0] == eid
        in_top5 = eid in top5_ids
        n_top1 += in_top1
        n_top5 += in_top5
        marker = "TOP1" if in_top1 else ("TOP5" if in_top5 else "MISS")
        print(f"[{marker}] {fname:25s} expected={eid}/{ename!r}")
        for i, (cid, name, p) in enumerate(zip(top5_ids, top5_names, top5_probs)):
            print(f"        {i+1}. {cid:4d}  {name:30s}  {p:.4f}")
    print()
    print(f"Top-1: {n_top1}/{len(EXPECTED)},  Top-5: {n_top5}/{len(EXPECTED)}")
    return 0 if n_top5 == len(EXPECTED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
