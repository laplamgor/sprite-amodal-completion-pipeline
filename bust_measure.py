"""Measure bust size with a purpose-trained detector.

The general VLM cannot do this axis: asked "how large is the bust" it answers
"small" or "medium" for every image, and its letter-softmax probabilities are
prior-dominated. Measured on the 35-image reference set, the best VLM question
reached 0.69 pairwise concordance against the hand ranking; this detector
reaches 0.91 (0.905 held out, 5-fold).

The model is Anzhc's "Breast size det cls v8" (YOLO11m, 15 classes, ~16k
annotated illustrations). Its classes are a booru/rule34 size scale:

    0 flat chest        5 huge breasts
    1 small breasts     6 gigantic breasts
    2 medium breasts    7..14 progressively absurd
    3 big breasts
    4 large breasts

Two passes are averaged: the whole figure at 960 px and the torso crop at
640 px. Averaging the two is what fixed the images whose chest is partly
hidden by an arm, a prop or loose clothing.

Writes `bust` (0..1, anime-calibrated) into input/axes/<stem>.json next to the
VLM axes, keeping the old value as `bust_vlm`.

    python bust_measure.py                # all of input/incoming
    python bust_measure.py --only 0297    # substring filter
    python bust_measure.py --report       # print the ranking, write nothing
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import config as C

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models", "breast_size_y11m.pt")
CONF = 0.20          # validated: 5-fold CV picked 0.20 in every fold
POWER = 4.0          # confidence sharpening for the weighted class mean

# class -> semantic position on 0..1, where 0.5 is the anime "average" (C/D cup)
LEVEL = {0: 0.05, 1: 0.22, 2: 0.38, 3: 0.55, 4: 0.72, 5: 0.88, 6: 1.00}
CLASS_NAME = {0: "flat chest", 1: "small breasts", 2: "medium breasts",
              3: "big breasts", 4: "large breasts", 5: "huge breasts",
              6: "gigantic breasts"}


def _load_bbox():
    """Alpha bounding box, 1%/99.5% trimmed so props do not stretch it."""
    def bbox(im):
        a = np.asarray(im)[:, :, 3] > 16
        if not a.any():
            return 0, 0, im.width, im.height
        rows = a.sum(1).astype(float)
        cum = np.cumsum(rows) / rows.sum()
        top = int(np.searchsorted(cum, 0.01))
        bot = int(np.searchsorted(cum, 0.995))
        cols = a.sum(0).astype(float)
        cum = np.cumsum(cols) / cols.sum()
        left = int(np.searchsorted(cum, 0.01))
        right = int(np.searchsorted(cum, 0.995))
        return left, top, max(right, left + 8), max(bot, top + 8)
    return bbox


def _square(rgb, side):
    s = side / max(rgb.size)
    rgb = rgb.resize((max(1, round(rgb.width * s)), max(1, round(rgb.height * s))),
                     Image.LANCZOS)
    if rgb.size == (side, side):
        return rgb
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return canvas


def views(path):
    """(full figure @960, torso crop @640) on white, as the model expects."""
    bbox = _load_bbox()
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    x0, y0, x1, y1 = bbox(im)
    h, w = y1 - y0, x1 - x0
    torso = rgb.crop((max(0, x0 - int(w * 0.05)), y0,
                      min(rgb.width, x1 + int(w * 0.05)), y0 + int(h * 0.55)))
    return _square(rgb, 960), _square(torso, 640)


def measure(model, path):
    """-> (score 0..1, class, confidence, per-view detections)"""
    dets = []
    for view, side in zip(views(path), (960, 640)):
        r = model.predict(view, imgsz=side, conf=CONF, verbose=False)[0]
        b = r.boxes
        dets.append([] if b is None or len(b) == 0 else
                    [{"cls": int(c), "conf": round(float(p), 3)}
                     for c, p in zip(b.cls.cpu().numpy(),
                                     b.conf.cpu().numpy())])
    vals = []
    for d in dets:
        if not d:
            vals.append(0.0)
            continue
        c = np.array([x["cls"] for x in d], float)
        p = np.array([x["conf"] for x in d], float)
        w = np.exp(p * POWER)
        vals.append(float((c * w).sum() / w.sum()))
    raw = sum(vals) / len(vals)                    # mean class index
    # interpolate the semantic level, then clamp to the known range
    lo = int(np.floor(raw))
    hi = min(6, lo + 1)
    frac = raw - lo
    score = LEVEL.get(lo, 1.0) * (1 - frac) + LEVEL.get(hi, 1.0) * frac
    conf = max([x["conf"] for d in dets for x in d], default=0.0)
    return round(score, 3), round(raw, 2), conf, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--report", action="store_true", help="print, do not write")
    ap.add_argument("--conf", type=float, default=CONF)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)

    files = [f for f in sorted(glob.glob(os.path.join(C.INCOMING, "*.png")))
             if not os.path.splitext(f)[0].endswith("_amodal")]
    if a.only:
        files = [f for f in files
                 if any(o.lower() in os.path.basename(f).lower() for o in a.only)]

    rows = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        score, raw, conf, dets = measure(model, f)
        rows.append((score, stem, raw, conf, dets))
        if not a.report:
            os.makedirs(C.BUST, exist_ok=True)
            with open(os.path.join(C.BUST, stem + ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"bust": score, "class": raw,
                           "conf": round(conf, 3),
                           "views": dets}, fh, indent=1)

    rows.sort(reverse=True)
    print("%-32s %6s %6s %6s  %s" % ("stem", "bust", "class", "conf", "views"))
    for score, stem, raw, conf, dets in rows:
        cls = CLASS_NAME.get(int(round(raw)), "?")
        v = " | ".join((" ".join("c%d@%.2f" % (d["cls"], d["conf"]) for d in dd)
                        or "-") for dd in dets)
        print("%-32s %6.3f %6.2f %6.2f  %-22s %s"
              % (stem[:32], score, raw, conf, cls, v))
    if not a.report:
        print("\nwrote %s\\<stem>.json" % C.BUST)
        print("next: python rederive.py --refit")


if __name__ == "__main__":
    main()
