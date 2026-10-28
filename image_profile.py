"""Estimate a body profile from ONE source image, deterministically.

Per image, never per character: two sprites of the same character (before and
after a growth spurt, a different outfit, a different pose) get separate
profiles, because the drawing is different. A profile is cached by image stem
only, so re-running the same file is byte-identical without re-measuring.

Measured from the image:
    skin / hair colour (segformer clothes parsing)
    silhouette proportions (alpha)
    chest relief, surface sharpness and lift (Depth Anything 3 map)

Derived, not measured (occluded by clothing, so invented - but seeded per
character, so the same character always gets the same answer):
    areola colour  = skin tone darkened toward red
    areola radius  = from bust size + character seed
    nipple projection = from firmness + character seed

Profiles are cached per image file, so a second image of the same character is
measured on its own rather than inheriting the first.

    python image_profile.py --dir incoming            # build/print the table
    python image_profile.py --dir incoming --force    # recompute
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

INPUT = C.INPUT
DEPTH_DIR = C.DEPTH
PROFILE_DIR = C.PROFILES
SEGFORMER = "mattmdjaga/segformer_b2_clothes"
CHAR_RE = re.compile(r"^([0-9]{3,4})_")

_cache = {}


# ---------------------------------------------------------------- segmentation
def _seg():
    if "m" not in _cache:
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        proc = SegformerImageProcessor.from_pretrained(SEGFORMER)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _cache["m"] = (proc, model)
    return _cache["m"]


def _labels(rgb_pil):
    proc, model = _seg()
    dev = next(model.parameters()).device
    inp = proc(images=rgb_pil, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        logits = model(**inp).logits
    logits = torch.nn.functional.interpolate(
        logits, size=(rgb_pil.height, rgb_pil.width), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


# ------------------------------------------------------------------- palettes
def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in rgb)


def _hsv(rgb):
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*(np.clip(rgb, 0, 255) / 255.0))
    return h * 360.0, s, v


def _skin_word(h, s, v):
    if v > 0.93 and s < 0.12:
        return "very fair skin"
    if v > 0.86 and s < 0.20:
        return "fair skin"
    if s < 0.28:
        return "light warm skin"
    if s < 0.38:
        return "warm tan skin"
    return "deep tan skin"


def _areola_word(rgb):
    h, s, v = _hsv(rgb)
    if v < 0.42:
        return "dark brown"
    if s < 0.22:
        return "pale pink"
    if s < 0.33:
        return "soft rose"
    if s < 0.45:
        return "rosy brown"
    return "dusky rose"


# ---------------------------------------------------------------- depth maths
def _depth_metrics(depth_png, alpha, box):
    d = np.array(Image.open(depth_png).convert("L")).astype(np.float32) / 255.0
    if d.shape != alpha.shape:
        d = np.array(Image.fromarray((d * 255).astype(np.uint8)).resize(
            (alpha.shape[1], alpha.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0

    # orient: body should read as "closer" than background
    outside = d[~alpha]
    inside = d[alpha]
    if outside.size and inside.size and outside.mean() < inside.mean():
        d = 1.0 - d

    y0, y1, x0, x1 = box
    h = y1 - y0
    band = np.zeros_like(alpha)
    band[y0 + int(h * 0.30):y0 + int(h * 0.43), x0:x1] = True
    m = band & alpha
    if m.sum() < 50:
        return None

    body = d[alpha]
    rng = float(np.percentile(body, 95) - np.percentile(body, 5)) or 1.0

    # bust bulge: how far the centre of the torso sits in front of its sides.
    # sampled per row so outstretched arms / props cannot inflate it.
    ys_all, xs_all = np.where(alpha)
    cx = int(np.median(xs_all))
    half = max(4, int(0.18 * (xs_all.max() - xs_all.min())))
    rows = []
    for y in range(y0 + int(h * 0.28), y0 + int(h * 0.42)):
        if y < 0 or y >= alpha.shape[0]:
            continue
        row = alpha[y]
        if not row[cx]:
            continue
        lx, rx = cx - half, cx + half
        if lx < 0 or rx >= alpha.shape[1] or not row[lx] or not row[rx]:
            continue
        sides = 0.5 * (d[y, lx] + d[y, rx])
        rows.append((sides - d[y, cx]) / rng)
    bulge = float(np.mean(rows)) if rows else 0.0

    chest = d[m]
    relief = float(np.percentile(chest, 92) - np.percentile(chest, 8)) / rng

    # surface sharpness: residual against a blurred copy
    blur = np.array(Image.fromarray((d * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(7)), dtype=np.float32) / 255.0
    sharp = float(np.abs(d[m] - blur[m]).mean()) / rng

    # lift: how high in the band the closest surface sits
    thr = np.percentile(chest, 80)
    ys = np.where(m)[0]
    top_y = ys.min()
    bh = max(1, ys.max() - top_y)
    lift = float(((d[m] >= thr) * (1.0 - (ys - top_y) / bh)).sum()) / max(1, int((d[m] >= thr).sum()))

    return {"relief": round(relief, 3), "sharp": round(sharp, 4), "lift": round(lift, 3),
            "bulge": round(bulge, 3)}


# ---------------------------------------------------------------- proportions
def _proportions(alpha, box):
    y0, y1, x0, x1 = box
    h = y1 - y0
    rows = []
    for f in (0.22, 0.35, 0.45, 0.55, 0.68):
        y = y0 + int(h * f)
        xs = np.where(alpha[max(0, y - 2):y + 3].any(axis=0))[0]
        rows.append(0.0 if xs.size == 0 else (xs.max() - xs.min()) / h)
    sh, chest, waist, hip, thigh = rows
    return {"shoulder": round(sh, 3), "chest": round(chest, 3), "waist": round(waist, 3),
            "hip": round(hip, 3), "thigh": round(thigh, 3)}


# ------------------------------------------------------------------ inference
def measure(src, depth_png):
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    alpha = np.array(im)[:, :, 3] > 128

    lab = _labels(rgb)
    arr = np.array(rgb).astype(np.float32)

    face_m = lab == 11
    skin_m = np.isin(lab, [11, 12, 13, 14, 15]) & alpha
    if face_m.sum() > 300:                      # prefer the face: never occluded
        skin_px = arr[face_m]
    else:
        px = arr[skin_m]
        if px.size:
            lum = px @ np.array([0.299, 0.587, 0.114])
            skin_px = px[lum <= np.percentile(lum, 85)]   # drop stockings / white cloth
        else:
            skin_px = arr[alpha]
    skin = np.median(skin_px, axis=0)

    hair_m = (lab == 2) & alpha
    hair = np.median(arr[hair_m], axis=0) if hair_m.sum() > 300 else skin * 0.6

    ys, xs = np.where(alpha)
    box = (ys.min(), ys.max(), xs.min(), xs.max())
    h = box[1] - box[0] + 1
    w = box[3] - box[2] + 1

    prop = _proportions(alpha, box)
    dep = _depth_metrics(depth_png, alpha, box) if depth_png else None

    skin_h, skin_s, skin_v = _hsv(skin)
    areola = np.clip(skin * np.array([0.74, 0.54, 0.52]), 0, 255)

    return {
        "source": os.path.basename(src),
        "skin": _hex(skin), "skin_word": _skin_word(skin_h, skin_s, skin_v),
        "hair": _hex(hair),
        "areola_color": _hex(areola), "areola_word": _areola_word(areola),
        "bbox": {"w": int(w), "h": int(h), "ratio": round(w / h, 3)},
        "proportions": prop,
        "depth": dep,
    }


def derive(prof, seed):
    """Fill the attributes the image cannot provide, deterministically per character."""
    rnd = random.Random(seed)
    dep = prof.get("depth") or {}
    prop = prof["proportions"]

    # bust size: forward bulge of the torso surface (pose-independent), clamped so a
    # stray prop or rigging cannot push it off the scale
    bulge = float(dep.get("bulge", 0.0))
    bulge = max(-0.2, min(0.8, bulge))
    if bulge < 0.12:
        size = "small"
    elif bulge < 0.45:
        size = "medium"
    else:
        size = "large"

    # shape: sharp local surface relief = taut/firm, smooth = soft
    sharp = dep.get("sharp", 0.1)
    shape = "firm" if sharp > 0.12 else "soft"

    # push-up: high lift + real bulge reads as supported/pressed
    lift = dep.get("lift", 0.4)
    push = "pressed together" if (lift > 0.50 and bulge > 0.30) else "relaxed"

    base_r = {"small": 20.0, "medium": 27.0, "large": 34.0}[size]
    radius_mm = base_r * rnd.uniform(0.85, 1.15)
    radius = "small" if radius_mm < 24 else ("medium" if radius_mm < 31 else "large")

    base_p = {"small": 1.0, "medium": 1.6, "large": 2.2}[size] * (1.15 if shape == "firm" else 0.9)
    proj_mm = base_p * rnd.uniform(0.8, 1.25)
    proj = "subtle" if proj_mm < 1.4 else ("moderate" if proj_mm < 2.2 else "prominent")

    ribs = "softly shaded" if shape == "soft" else "gently defined"

    prof["bust"] = {"size": size, "shape": shape, "push_up": push,
                    "bulge": round(bulge, 3)}
    prof["areola"] = {"radius": radius, "radius_mm": round(radius_mm, 1)}
    prof["nipple"] = {"projection": proj, "projection_mm": round(proj_mm, 2)}
    prof["ribs"] = ribs
    prof["seed"] = seed
    return prof


def build_prompt(p):
    """Natural sentences, not tag soup - Qwen follows prose better."""
    b, a, n = p["bust"], p["areola"], p["nipple"]
    return (" Her body: {sz} {sh} breasts that sit {pu}, {ac} areolas of {ar} size, "
            "{np} nipples, {rib} ribcage, {sk}.").format(
        sz=b["size"], sh=b["shape"], pu=b["push_up"], ac=p["areola_word"],
        ar=a["radius"], np=n["projection"], rib=p["ribs"], sk=p["skin_word"])


def char_of(stem):
    m = CHAR_RE.match(stem)
    return m.group(1) if m else stem.split("_")[0]


def profile_for(src, force=False):
    stem = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = os.path.join(PROFILE_DIR, stem + ".json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    depth_png = os.path.join(DEPTH_DIR, stem + ".png")
    prof = measure(src, depth_png if os.path.exists(depth_png) else None)
    prof["image"] = stem
    seed = int(hashlib.blake2b(stem.encode(), digest_size=6).hexdigest(), 16)
    prof = derive(prof, seed)
    prof["prompt"] = build_prompt(prof)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1)
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="incoming")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    base = os.path.join(INPUT, a.dir)
    files = sorted(glob.glob(os.path.join(base, "*.png")))
    files = [f for f in files if not os.path.splitext(f)[0].endswith("_amodal")]

    hdr = ("image", "skin", "hair", "bust", "shape", "push", "areola", "nipple", "ribs", "depth")
    print("%-14s %-9s %-9s %-7s %-6s %-17s %-12s %-10s %-15s %s" % hdr)
    for f in files:
        p = profile_for(f, a.force)
        d = p.get("depth") or {}
        print("%-14s %-9s %-9s %-7s %-6s %-17s %-12s %-10s %-15s relief=%.2f sharp=%.3f" % (
            p["image"][:14], p["skin"], p["hair"], p["bust"]["size"], p["bust"]["shape"],
            p["bust"]["push_up"], p["areola_word"] + "/" + p["areola"]["radius"],
            p["nipple"]["projection"], p["ribs"], d.get("relief", 0), d.get("sharp", 0)))

    print("\nprompt fragments (one per image):")
    for f in files:
        p = profile_for(f)
        print("  %-14s ->%s" % (p["image"][:14], p["prompt"]))


if __name__ == "__main__":
    main()
