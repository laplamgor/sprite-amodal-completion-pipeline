"""Per-prompt SAM3 mask area, exactly as run_folder.branch() builds it.

Diagnostic only: no sampler, no VAE decode.  Writes one mask per prompt and
prints its share of the canvas, so a protection mask that swallows the whole
garment is visible instead of inferred from a union.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import comfy_client as CC  # noqa: E402
import run_folder as R  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

PROMPTS = [
    ("outfit", "outfit"),
    ("garments", R.GARMENTS),
    ("face", "face, mouth, lips"),
    ("hair", "hair, braid"),
    ("hands", "hand, fingers"),
    ("footwear", "shoe, boot, footwear, sandal"),
    ("equipment", R.EQUIPMENT),
]


def probe(stem, rel, thr, only=None):
    """One prompt per word: a single graph hides errors behind partial output."""
    for i, (name, text) in enumerate(PROMPTS):
        if only and name != only:
            continue
        wf = {
            "900": {"class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": R.SAM}},
            "1": {"class_type": "LoadImage", "inputs": {"image": rel}},
            "2": {"class_type": "AmodalSafeFrame", "inputs": {
                "image": ["1", 0], "mask": ["1", 1],
                "margin": 0.08, "scale": 1.25, "max_megapixels": 1.5,
                "min_aspect": 0.62, "max_aspect": 1.60}},
            "10": {"class_type": "CLIPTextEncode",
                   "inputs": {"text": text, "clip": ["900", 1]}},
            "11": {"class_type": "SAM3_Detect", "inputs": {
                "model": ["900", 0], "image": ["2", 0],
                "conditioning": ["10", 0],
                "threshold": thr, "refine_iterations": 2,
                "individual_masks": False}},
            "12": {"class_type": "MaskToImage", "inputs": {"mask": ["11", 0]}},
            "13": {"class_type": "SaveImage", "inputs": {
                "images": ["12", 0],
                "filename_prefix": "probe_%s_%s_%s" % (stem[:12], name, thr)}},
        }
        pid = CC.post_prompt(wf)["prompt_id"]
        CC.wait_for(pid, timeout=900)


def areas(stem, thr):
    out = {}
    for name, _ in PROMPTS:
        g = [f for f in os.listdir(C.OUTPUT)
             if f.startswith("probe_%s_%s_%s" % (stem[:12], name, thr))]
        if not g:
            out[name] = None
            continue
        a = np.asarray(Image.open(os.path.join(C.OUTPUT, g[0])).convert("L"))
        out[name] = round(100.0 * (a > 127).mean(), 1)
        os.remove(os.path.join(C.OUTPUT, g[0]))
    return out


if __name__ == "__main__":
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    stems = sys.argv[2:]
    print("%-26s %s" % ("stem", "  ".join("%-10s" % p[0] for p in PROMPTS)))
    for s in stems:
        rel = s + ".png" if s.endswith(".png") is False else s
        probe(os.path.splitext(s)[0], rel, thr)
        a = areas(os.path.splitext(s)[0], thr)
        print("%-26s %s" % (s[:26], "  ".join(
            "%-10s" % ("-" if a[n] is None else "%.1f%%" % a[n]) for n, _ in PROMPTS)))
