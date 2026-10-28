"""Compare every source PNG against its rendered output and build contact sheets.

    python evaluate.py                 # table + contact sheets in ComfyUI/output
    python evaluate.py --no-sheets     # table only
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

import numpy as np
from PIL import Image, ImageDraw

IN = C.INCOMING
OUT = C.OUTPUT
HEIGHT = 430
PER_SHEET = 4


def flatten(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def alpha(p):
    return np.array(Image.open(p).convert("RGBA"))[:, :, 3] > 127


def output_for(stem):
    """Newest <stem>_amodal*.png: ComfyUI appends _00001_ when it cannot
    overwrite, and the rename back can lose that race."""
    exact = os.path.join(OUT, stem + "_amodal.png")
    if os.path.exists(exact):
        return exact
    hits = glob.glob(os.path.join(OUT, stem + "_amodal*.png"))
    return max(hits, key=os.path.getmtime) if hits else None


def measure(files):
    rows = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        o = output_for(stem)
        if o is None:
            rows.append((stem, None, None, None))
            continue
        a, b = alpha(f), alpha(o)
        iou = (a & b).sum() / max(1, (a | b).sum())
        src = np.array(Image.open(f).convert("RGBA")).astype(np.float32) / 255.0
        out = np.array(Image.open(o).convert("RGBA")).astype(np.float32) / 255.0
        d = np.abs(out[:, :, :3] - src[:, :, :3]).mean(axis=2)
        rows.append((stem, iou, b.sum() / a.sum() - 1, d[a].mean()))
    return rows


def sheets(files):
    for page in range(0, len(files), PER_SHEET):
        cells = []
        for f in files[page:page + PER_SHEET]:
            stem = os.path.splitext(os.path.basename(f))[0]
            o = output_for(stem)
            if o is None:
                continue

            def rs(p):
                im = flatten(p)
                return im.resize((int(im.width * HEIGHT / im.height), HEIGHT),
                                 Image.LANCZOS)

            cells.append((stem[:10], rs(f), rs(o)))
        if not cells:
            continue
        w = max(sum(a.width + b.width + 26 for _, a, b in cells), 10)
        canvas = Image.new("RGB", (w, HEIGHT + 22), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, a, b in cells:
            canvas.paste(a, (x, 22))
            x += a.width + 4
            canvas.paste(b, (x, 22))
            x += b.width + 22
            draw.text((x - b.width - 20), label + "   src | v19", fill=(0, 0, 0))
        path = os.path.join(OUT, "sheet_%d.png" % (page // PER_SHEET + 1))
        canvas.save(path)
        print("wrote", path, canvas.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(IN, "*.png")))
    rows = measure(files)
    print("%-40s %6s %8s %7s" % ("stem", "IoU", "dAlpha", "diffIn"))
    for stem, iou, da, di in rows:
        if iou is None:
            print("%-40s  MISSING" % stem)
        else:
            print("%-40s %6.3f %7.1f%% %7.3f" % (stem, iou, da * 100, di))
    if not a.no_sheets:
        sheets(files)


if __name__ == "__main__":
    main()
