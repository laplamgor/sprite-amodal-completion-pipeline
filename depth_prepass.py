"""Pre-pass: Depth Anything 3 over a folder, saved as PNGs next to the sources.

Runs the depth model *before* the sampler so it never competes for VRAM with the
20B edit model. Output: ComfyUI/input/depth/<stem>.png (16-bit-safe greyscale).

    python depth_prepass.py --dir incoming
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C           # noqa: E402
import comfy_client as t  # noqa: E402
import run_folder as rf      # noqa: E402

DEPTH_DIR = C.DEPTH
DEPTH_MODEL = "depth_anything_3_base.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="incoming")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resolution", type=int, default=504)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(DEPTH_DIR, exist_ok=True)
    files = rf.discover(a.dir)
    if a.only:
        files = [f for f in files if any(o.lower() in f[0].lower() for o in a.only)]

    todo = []
    for rel, stem in files:
        out = os.path.join(DEPTH_DIR, stem[:40] + ".png")
        if not a.force and os.path.exists(out):
            continue
        todo.append((rel, stem[:40]))
    print(f"{len(files)} images, {len(todo)} need depth")
    if not todo:
        return

    wf = {"900": {"class_type": "LoadDA3Model",
                  "inputs": {"model_name": DEPTH_MODEL, "weight_dtype": "default"}}}
    nid = 1000
    for rel, stem in todo:
        wf[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": rel}}
        wf[str(nid + 1)] = {"class_type": "DA3Inference", "inputs": {
            "da3_model": ["900", 0], "image": [str(nid), 0],
            "resolution": a.resolution, "resize_method": "upper_bound_resize",
            "mode": "mono"}}
        wf[str(nid + 2)] = {"class_type": "DA3Render", "inputs": {
            "da3_geometry": [str(nid + 1), 0], "output": "depth",
            "output.normalization": "min_max", "output.apply_sky_clip": False}}
        wf[str(nid + 3)] = {"class_type": "SaveImage", "inputs": {
            "images": [str(nid + 2), 0], "filename_prefix": "depth_" + stem}}
        nid += 100

    t0 = time.time()
    r = t.post_prompt(wf)
    print("submitted", r["prompt_id"], r.get("node_errors"))
    h = t.wait_for(r["prompt_id"], timeout=7200)
    print("status", h["status"]["status_str"], f"{time.time()-t0:.0f}s")

    import re
    import shutil
    out_root = os.path.join(BASE, "ComfyUI", "output")
    moved = 0
    for k, d in h.get("outputs", {}).items():
        for im in d.get("images", []):
            src = os.path.join(out_root, im.get("subfolder", ""), im["filename"])
            stem = re.sub(r"_\d+_\.png$", "", im["filename"])
            stem = re.sub(r"^depth_", "", stem)
            dst = os.path.join(DEPTH_DIR, stem + ".png")
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1
    print(f"depth maps written: {moved} -> {DEPTH_DIR}")


if __name__ == "__main__":
    main()
