"""12 forced-choice axes -> 17 derived anatomy terms -> prompt fragment.

Elicitation: Qwen2.5-VL-3B, softmax over the answer letters.
Position bias: every axis is asked in K shuffled option orders, averaged back
into canonical order.

`build` is asked as two concrete questions and averaged (see BUILD_PAIR); the
abstract "how is her figure drawn overall" wording scored 0.53 against the hand
rank, the pair scores 0.70 (_eval/FINDINGS.md section 7).

The vision tower encodes the image once; the 12 axes x K orders (36 rows) are
then scored in slices of ROWS so the peak allocation stays flat. Every image is
letterboxed to SIDE x SIDE first, which keeps the token count - and therefore
the activation shapes - identical for every image.

    python vlm_axes.py                 # all sources
    python vlm_axes.py --only 1005     # substring filter
    python vlm_axes.py --rows 7        # less VRAM, same result
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from copy import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# must be set before torch initialises CUDA; stops the allocator hoarding
# differently-shaped blocks until the card spills into shared memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import config as C  # noqa: E402

IN = C.INCOMING
OUT = C.AXES
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LETTERS = "ABCDE"
SHUFFLES = 3
SIDE = 512                  # every image is letterboxed to SIDE x SIDE
MAX_PIXELS = SIDE * SIDE    # vision tokens are the cost driver; 512^2 is plenty here
ROWS = 14                   # rows per forward pass (36 rows total)

# id, question, options low -> high
#
# `build` used to be one abstract question ("How is her figure drawn overall?")
# and it ranked badly: 0.53 against the hand rank, with everything below the
# top bin collapsed together. Two concrete, reference-anchored questions
# average to 0.70. Keep them as separate rows -- they are combined after
# scoring, not asked as one question.
BUILD_PAIR = ("build_mass", "build_arm")

AXES = [
    ("build_mass", "Compared with a typical anime heroine, is this character "
     "drawn thinner or heavier? Judge body mass only, not height.",
     ["much thinner than average", "thinner than average", "about average",
      "heavier than average", "much heavier than average"]),
    ("build_arm", "How thick is her upper arm drawn, compared with the width "
     "of her head?",
     ["far narrower than her head", "narrower than her head",
      "about half her head width", "about as wide as her head",
      "wider than her head"]),
    ("waist_taper", "Does her waist narrow between ribcage and hips?",
     ["straight, no taper", "slight taper", "clear taper", "strong hourglass taper"]),
    ("bust", "How much chest volume is under the clothing?",
     ["flat", "small", "moderate", "large", "very large"]),
    ("exposure", "How much skin is shown on the torso, shoulders and legs?",
     ["fully covered", "arms and lower legs", "some skin", "bare midriff"]),
    ("posture", "How is her body held?",
     ["relaxed slouch", "natural", "upright", "taut and posed"]),
    ("muscle", "Does the drawing suggest muscle tone?",
     ["soft, untoned", "slight tone", "athletic", "clearly muscular"]),
    ("proportion", "What are her body proportions?",
     ["chibi", "short", "standard", "tall", "long-legged and slim"]),
    ("line", "Describe the line work.",
     ["soft, barely visible lines", "thin clean lines", "medium lines",
      "bold inked outlines"]),
    ("shade", "Describe the shading.",
     ["flat cel shading", "cel with soft gradients", "soft painterly",
      "heavy airbrush"]),
    ("gloss", "Describe the skin finish.",
     ["matte", "slight sheen", "glossy highlights", "very glossy"]),
    ("detail", "How detailed is the rendering?",
     ["minimalist", "moderate", "detailed", "dense hatching and texture"]),
]

# Dropped on the way here, both measured rather than assumed:
#   limb     never read by DERIVED / STYLE / FAMILY, 0/35 prompts, dead weight
#   age_imp  16/35 images scored under 0.25 ("childlike") and 31/35 under 0.40,
#            so the calibration stretched noise across 27/35 prompts.  The
#            adult assertion lives in comfy_client.POS_TXT, not here.
#   hip_bias 30/35 images in one bin, 150 testable pairs, no usable
#            resolution; the "shoulder frame" term went with it.

# derived: name, formula over the axis dict, phrase when high, phrase when low
DERIVED = [
    ("visible ribs",
     lambda a: 0.55 * (1 - a["build"]) + 0.25 * a["muscle"] - 0.45 * a["bust"],
     "visible ribs", "smooth skin over the ribcage"),
    ("thoracic arch",
     lambda a: 0.45 * (1 - a["build"]) + 0.30 * a["waist_taper"],
     "wide thoracic arch", "narrow ribcage"),
    ("ribcage contour",
     lambda a: 0.50 * (1 - a["build"]) + 0.30 * a["line"] - 0.30 * a["bust"],
     "sharply drawn ribcage contour", "soft ribcage contour"),
    ("ab definition",
     lambda a: 0.65 * a["muscle"] + 0.20 * (1 - a["build"]) - 0.30 * a["bust"],
     "defined abs", "smooth abdomen"),
    ("flat abdomen",
     lambda a: 0.40 * (1 - a["build"]) + 0.25 * a["posture"],
     "flat stomach", None),
    ("obliques",
     lambda a: 0.60 * a["muscle"] + 0.25 * a["waist_taper"],
     "chiselled obliques", "soft flanks"),
    ("soft belly",
     lambda a: 0.60 * a["build"] + 0.25 * (1 - a["muscle"]),
     "soft belly", "flat stomach"),
    ("linea alba",
     lambda a: 0.50 * a["muscle"] + 0.30 * (1 - a["build"]),
     "visible linea alba", "no centre line"),
    ("hip bones",
     lambda a: 0.55 * (1 - a["build"]),
     "visible hip bones", "smooth hips"),
    ("collarbone",
     lambda a: 0.50 * (1 - a["build"]) + 0.30 * a["exposure"] + 0.20 * a["proportion"],
     "prominent collarbones", "smooth collarbone line"),
    ("inframammary fold",
     lambda a: 0.65 * a["bust"] + 0.20 * (1 - a["build"]),
     "deep infra-mammary fold", "smooth underbust"),
    ("underbust shadow",
     lambda a: 0.55 * a["bust"] + 0.20 * a["shade"],
     "strong underbust shadow", "even underbust lighting"),
    ("serratus",
     lambda a: 0.70 * a["muscle"] - 0.25 * a["bust"],
     "visible serratus", "smooth flanks"),
    ("navel depth",
     lambda a: 0.40 * (1 - a["build"]) + 0.30 * a["bust"] + 0.30 * a["shade"],
     "deep navel", "shallow navel"),
    ("waist definition",
     lambda a: 0.60 * a["waist_taper"],
     "cinched waist", "straight waist"),
    ("thigh gap",
     lambda a: 0.50 * (1 - a["build"]) + 0.30 * a["proportion"],
     "thigh gap", "touching thighs"),
    ("skin tension",
     lambda a: 0.50 * (1 - a["build"]) + 0.30 * a["muscle"] + 0.20 * a["gloss"],
     "taut skin", "soft skin"),
]

# opposing pairs: the weaker one is dropped
PAIRS = [("visible ribs", "soft belly"), ("ab definition", "soft belly"),
         ("flat abdomen", "soft belly"), ("visible ribs", "flat abdomen")]

STYLE = [("line", "delicate thin lineart", "bold inked outlines"),
         ("shade", "flat cel shading", "soft airbrushed gradients"),
         ("gloss", "matte skin", "glossy specular highlights"),
         ("detail", "minimal rendering", "dense hatching texture")]


def load():
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"      # so logits[:, -1] is each row's last token
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def letter_ids(tok, n):
    return [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS[:n]]


def build_jobs():
    """Flat list of (axis id, canonical order, prompt) for all axes x orders."""
    jobs, meta = [], {}
    for aid, question, options in AXES:
        n = len(options)
        rnd = random.Random(aid)                  # same orders for every image
        orders = [list(range(n))]
        for _ in range(SHUFFLES - 1):
            o = list(range(n))
            rnd.shuffle(o)
            orders.append(o)
        meta[aid] = {"n": n, "orders": orders}
        for order in orders:
            shown = [options[i] for i in order]
            jobs.append((aid, order,
                         question + " Choose one:\n"
                         + "\n".join("%s. %s" % (LETTERS[k], shown[k])
                                     for k in range(n))
                         + "\nAnswer with one letter only."))
    return jobs, meta


def prep_image(path, side=SIDE):
    """Composite on white and letterbox to a fixed square.

    A constant pixel size means a constant token count, so every image produces
    the exact same activation shapes and the CUDA allocator can reuse its blocks.
    Variable shapes were what made reserved memory creep up until Windows started
    paging the model out to shared GPU memory.
    """
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    w, h = rgb.size
    scale = side / max(w, h)
    rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.LANCZOS)
    if rgb.size == (side, side):
        return rgb
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return canvas


@torch.no_grad()
def score_all(proc, model, img, jobs, meta, ids_by_axis, rows=ROWS):
    """Score every axis+order for one image, reusing the vision features.

    The image is encoded once; the rows are then run in slices of `rows` so the
    peak allocation stays flat regardless of how many axes there are.
    """
    texts = [proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": q}]}],
        tokenize=False, add_generation_prompt=True) for _, _, q in jobs]
    kw = {"text": texts, "images": [img] * len(texts),
          "return_tensors": "pt", "padding": True}
    try:
        inp = proc(max_pixels=MAX_PIXELS, **kw)
    except TypeError:
        inp = proc(**kw)

    # Vision tower once: the processor emitted one patch block per copy, but
    # only the first is encoded. Its features are reused for every row.
    n0 = int(inp["image_grid_thw"][0].prod())
    feat = model.get_image_features(inp["pixel_values"][:n0].to("cuda"),
                                    inp["image_grid_thw"][:1].to("cuda"),
                                    return_dict=True)
    del inp["pixel_values"]

    rows = max(1, min(rows, len(texts)))
    acc = {aid: [0.0] * meta[aid]["n"] for aid in meta}
    for a in range(0, len(texts), rows):
        b = min(a + rows, len(texts))
        chunk = {k: (v[a:b].to("cuda") if torch.is_tensor(v) else v)
                 for k, v in inp.items()}
        sub = copy(feat)
        sub.pooler_output = tuple(feat.pooler_output) * (b - a)
        # use_cache=False: a single forward needs no KV cache, and it was
        # holding ~0.5 GB alive for the whole pass
        out = model(**chunk, mm_encoder_outputs={"image": sub},
                    logits_to_keep=1, use_cache=False)
        logits = out.logits[:, -1, :]
        for i in range(b - a):
            aid, order, _q = jobs[a + i]
            p = torch.softmax(logits[i][ids_by_axis[aid]].float(),
                              dim=0).cpu().tolist()
            for k, canonical in enumerate(order):
                acc[aid][canonical] += p[k] / len(meta[aid]["orders"])
        del out, chunk, sub, logits
    del feat
    axes = {aid: round(scalar(acc[aid]), 3) for aid in meta}
    # the two concrete build questions average into the single axis every
    # formula downstream expects
    axes["build"] = round(
        sum(axes[aid] for aid in BUILD_PAIR) / len(BUILD_PAIR), 3)
    return axes


def scalar(p):
    n = len(p)
    return sum(v * i / (n - 1) for i, v in enumerate(p))


def render(axes, keep=6):
    vals = {}
    for name, fn, hi, lo in DERIVED:
        v = max(0.0, min(1.0, fn(axes)))
        vals[name] = v

    for a, b in PAIRS:
        if a in vals and b in vals:
            if abs(vals[a] - 0.5) >= abs(vals[b] - 0.5):
                vals[b] = 0.5
            else:
                vals[a] = 0.5

    items = [(n, v) for n, v in vals.items() if abs(v - 0.5) >= 0.18]
    items.sort(key=lambda kv: -abs(kv[1] - 0.5))
    items = items[:keep]

    phrase = {n: (hi, lo) for n, _, hi, lo in DERIVED}
    parts = []
    for n, v in items:
        hi, lo = phrase[n]
        wob = "" if abs(v - 0.5) < 0.32 else "very "
        parts.append(wob + (hi if v > 0.5 else lo))

    style = []
    for aid, lo_txt, hi_txt in STYLE:
        v = axes[aid]
        if abs(v - 0.5) >= 0.18:
            style.append(hi_txt if v > 0.5 else lo_txt)
    return vals, parts, style[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=ROWS,
                    help="rows per forward pass (lower = less VRAM)")
    ap.add_argument("--side", type=int, default=SIDE,
                    help="letterbox size; every image is padded to side x side")
    a = ap.parse_args()

    proc, model = load()
    jobs, meta = build_jobs()
    ids_by_axis = {aid: letter_ids(proc.tokenizer, meta[aid]["n"])
                   for aid in meta}
    files = [f for f in sorted(glob.glob(os.path.join(IN, "*.png")))
             if not os.path.splitext(f)[0].endswith("_amodal")]
    if a.only:
        files = [f for f in files
                 if any(o.lower() in os.path.basename(f).lower() for o in a.only)]

    os.makedirs(OUT, exist_ok=True)
    results = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        cached = os.path.join(OUT, stem + ".json")
        if os.path.exists(cached):
            try:
                with open(cached, encoding="utf-8") as fh:
                    old = json.load(fh)["axes"]
                # a cache written by an older axis set is not reusable
                if all(k in old for k in ["build"] + [a for a, _, _ in AXES]):
                    results[stem] = old
                    print(stem[:20], "cached")
                    continue
                print(stem[:20], "stale cache, rescoring")
            except (KeyError, ValueError):
                pass                      # truncated file from an earlier kill
        rgb = prep_image(f, a.side)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        axes = score_all(proc, model, rgb, jobs, meta, ids_by_axis, a.rows)
        results[stem] = axes
        # write straight away so a kill keeps everything already scored
        with open(cached, "w", encoding="utf-8") as fh:
            json.dump({"axes": axes}, fh, indent=1)
        print("%-20s %5.1fs  peak %.0f MB  %s" % (
            stem[:20], time.time() - t0,
            torch.cuda.max_memory_allocated() / 1e6, json.dumps(axes)))
        del rgb
        torch.cuda.empty_cache()

    print("\n%-16s %s" % ("axis", " ".join("%-7s" % s[:6] for s in results)))
    for aid in [a for a, _, _ in AXES] + ["build"]:
        print("%-16s %s" % (aid, " ".join("%-7.2f" % r[aid] for r in results.values())))

    print("\nderived + prompt")
    for stem, axes in results.items():
        vals, parts, style = render(axes)
        print("\n%s" % stem[:20])
        print("  anatomy:", ", ".join(parts) if parts else "(none above threshold)")
        print("  style  :", ", ".join(style) if style else "(none)")
        with open(os.path.join(OUT, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"axes": axes, "derived": vals}, fh, indent=1)

    print("\nNOTE: these are the raw, uncalibrated phrases. Run rederive.py to "
          "write the calibrated prompt that the batch actually uses.")


if __name__ == "__main__":
    main()
