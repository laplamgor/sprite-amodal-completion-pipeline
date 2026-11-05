"""Per-word SAM3 mask area for one image, one prompt per word.

Used to prune the EQUIPMENT protection list: a protection mask is subtracted
from the clothing mask, so a word that fires on a garment silently turns the
inpaint off (0932's latex suit, 0722's dress, 0547's cloak -- all 22-34% canvas
from the *union*, which is why the union has to be split per word).

    python _eval/probe_words.py 0.25 0722_7804_hbbpfodwvfop_v66 \
        "weapon" "gun" "sword" "shield" "backpack" "bag" "machinery" "equipment"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import comfy_client as CC  # noqa: E402
import run_folder as R  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def run(stem, rel, text, thr):
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
            "model": ["900", 0], "image": ["2", 0], "conditioning": ["10", 0],
            "threshold": thr, "refine_iterations": 2,
            "individual_masks": False}},
        "12": {"class_type": "MaskToImage", "inputs": {"mask": ["11", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {
            "images": ["12", 0],
            "filename_prefix": "probe_%s_%s_%s" % (stem[:12], text, thr)}},
    }
    CC.wait_for(CC.post_prompt(wf)["prompt_id"], timeout=900)
    hits = [f for f in os.listdir(C.OUTPUT)
            if f.startswith("probe_%s_" % stem[:12]) and ("_%s_" % text) in f]
    if not hits:
        return None
    a = np.asarray(Image.open(os.path.join(C.OUTPUT, sorted(hits)[0])).convert("L"))
    for f in hits:
        os.remove(os.path.join(C.OUTPUT, f))
    return round(100.0 * (a > 127).mean(), 1)


if __name__ == "__main__":
    thr = float(sys.argv[1])
    stem = sys.argv[2]
    words = sys.argv[3:]
    base = os.path.splitext(stem)[0]
    print("%-26s %s" % ("stem", "  ".join("%-10s" % w for w in words)))
    vals = [run(base, base + ".png", w, thr) for w in words]
    print("%-26s %s" % (base[:26], "  ".join(
        "%-10s" % ("-" if v is None else "%.1f%%" % v) for v in vals)))
