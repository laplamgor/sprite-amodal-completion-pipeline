"""View label: is her torso turned away from the camera?

The failure this exists for: the inpaint prompt asserts front anatomy ("bare
torso, bare breasts"), so a back-facing source is rebuilt as a front and the arms
are re-synthesised with it.  0479 is the worked example (torso away from the
camera, head over the shoulder, long sleeves).

## The gate is the tagger alone

WD SwinV2 v3 (`models/wd-swinv2-tagger-v3.onnx`, 446 MB, CPU, ~0.6 s/image),
floor `from_behind >= 0.01`. Hand labels are the operator's; three passes were
needed, two of them mine and wrong, so the table is the record:

| image | `from_behind` | `breasts` | truth |
|---|---|---|---|
| 0228 | 0.597 | 0.020 | back |
| 0313 | 0.488 | 0.007 | back |
| 0941 | 0.462 | 0.790 | back |
| 0584 | 0.436 | 0.005 | back |
| 0450 | 0.374 | 0.783 | back |
| 0878 | 0.263 | 0.923 | back (3/4, chest edge) |
| 0479 | 0.083 | 0.021 | back |
| 0958 | 0.019 | 0.072 | back (3/4) |
| 0390 | 0.011 | 0.005 | mixed (upper back, hips front) |
| every other front | <= 0.003 | | front |

8 of 8 backs and no front, plus the mixed one at 0.011 -- that is the whole margin
between the weakest back and the strongest front, so the floor is fitted to this
corpus and should be re-read when a new batch lands.

`breasts` used to be a veto (`< 0.20`). It existed only to suppress 0878, which
was mislabelled as a front at the time; with 0878 known to be a 3/4 back, the
veto was deleting a true positive. Do not reintroduce a chest term.

## The VLM is recorded, not consulted

Qwen3-VL-4B (and 8B) were the second opinion for a while. On the 27-image corpus
the 4B finds 5 of the 8 backs -- 0313, 0450 and 0941 are read as chests -- and
its one "back" call on a front (0653) is already rejected by the tagger's floor.
8B scores the same 5 of 8. Thinking mode answered identically at ~5x the tokens
and sometimes did not finish.

So the conjunction was strictly worse than the tagger: 5/8 instead of 8/8, with
nothing gained on the front side. `--vlm` still asks it and writes the answer
into the cache, but the answer no longer changes `view`.

Writes `output/axes/_views.json`:

    {"view": "back", "candidate": true, "from_behind": 0.083, "breasts": 0.021, "vlm": null}

    python view_axis.py                 # all sources
    python view_axis.py --only 0479     # substring filter
    python view_axis.py --vlm           # also record the 4B's answer (~8.6 GB)
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

IN = C.INPUT
OUT = C.AXES
CACHE = os.path.join(OUT, "_views.json")

TAGGER = os.path.join(C.MODELS, "wd-swinv2-tagger-v3.onnx")
# The column order must be the model's own: this tagger emits 10861 scores, and
# the v1-4 (9083) tag list silently mislabels every axis after the first few.
TAGS = os.path.join(C.MODELS, "wd-v3-selected-tags.csv")
SIDE = 448                  # the tagger's fixed input

BEHIND_MIN = 0.01           # "any evidence of a back"; 0653 (front) scores 0.002
# NB: there used to be a `breasts >= 0.20` veto here. It existed to suppress
# 0878, which was mislabelled as a front at the time; 0878 is a 3/4 back and the
# veto was only deleting a true positive. Do not reintroduce a chest term.

VLM = "Qwen/Qwen3-VL-4B-Instruct"
QUESTION = ("Look at her torso only. Do we see her chest, or her back? "
            "Answer in five words.")
VLM_PIXELS = 512 * 512      # same letterbox as vlm_axes


def load_tags(path):
    """tag name -> output column, in file order (the ONNX output order)."""
    names = []
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for row in csv.reader(fh):
            if len(row) >= 3:
                names.append(row[1])
    return {n: i for i, n in enumerate(names)}


def load_tagger():
    import onnxruntime as ort
    # CPU: ~0.6 s/image, and it must not compete with the 20B model for VRAM.
    sess = ort.InferenceSession(TAGGER, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, load_tags(TAGS)


def load_vlm():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(VLM)
    model = AutoModelForImageTextToText.from_pretrained(
        VLM, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def prep_image(path, side=SIDE):
    """Composite on white, pad to a square, BGR -- the tagger's training input.

    The sources are transparent full-body illustrations; compositing on white is
    what the tagger saw at training time.  Padding rather than cropping keeps the
    whole figure, which is what the view label depends on.
    """
    import cv2
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    a = np.asarray(bg.convert("RGB"))
    h, w = a.shape[:2]
    pad = abs(h - w)
    if h > w:
        a = np.pad(a, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=255)
    else:
        a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), constant_values=255)
    a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return a.astype(np.float32)[None]


def score(sess, input_name, index, path):
    p = sess.run(None, {input_name: prep_image(path)})[0][0]
    return (float(p[index["from_behind"]]), float(p[index["breasts"]]))


def candidate(behind, breasts):
    """Tagger half of the gate: any evidence of a back.

    `breasts` is kept in the signature because it is written to the cache, but it
    is no longer a condition -- see the note at BEHIND_MIN.
    """
    return behind >= BEHIND_MIN


def ask_vlm(proc, model, path):
    """Generated answer to the view question; returns (label, answer)."""
    import torch
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    w, h = rgb.size
    scale = SIDE / max(w, h)
    rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (SIDE, SIDE), (255, 255, 255))
    canvas.paste(rgb, ((SIDE - rgb.width) // 2, (SIDE - rgb.height) // 2))

    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": QUESTION}]}],
        tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[canvas], return_tensors="pt")
    inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    ans = " ".join(proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                         skip_special_tokens=True).split())
    low = ans.lower()
    ib = low.rfind("back")
    ic = max(low.rfind("chest"), low.rfind("cleavage"), low.rfind("breast"))
    label = "unknown" if (ib < 0 and ic < 0) else ("back" if ib > ic else "front")
    return label, ans


def load_cache():
    if os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as fh:
                return json.load(fh)
        except ValueError:
            return {}
    return {}


def write_cache(cache):
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    os.replace(tmp, CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vlm", action="store_true",
                    help="also ask Qwen3-VL-4B and record its answer as advice "
                         "(does not change the label; it loses 3 of 8 backs)")
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(IN, "*.png")))
             if not os.path.splitext(f)[0].endswith("_amodal")]
    if a.only:
        files = [f for f in files
                 if any(o.lower() in os.path.basename(f).lower() for o in a.only)]

    cache = load_cache()
    todo = [(f, os.path.splitext(os.path.basename(f))[0]) for f in files
            if a.force or os.path.splitext(os.path.basename(f))[0] not in cache]
    if not todo:
        print("nothing to score")
        return

    tagger = load_tagger()
    vlm = None
    for f, stem in todo:
        t0 = time.time()
        behind, breasts = score(tagger[0], tagger[1], tagger[2], f)
        cand = candidate(behind, breasts)
        rec = {"from_behind": round(behind, 3), "breasts": round(breasts, 3),
               "candidate": cand,
               # The tagger is the gate. The 4B is recorded, never consulted:
               # over 27 images it found 5 of 8 backs and added no precision.
               "view": "back" if cand else "front", "vlm": None}
        if cand and a.vlm:
            if vlm is None:
                vlm = load_vlm()
            label, ans = ask_vlm(vlm[0], vlm[1], f)
            rec["vlm"] = label
            rec["answer"] = ans
        cache[stem] = rec
        write_cache(cache)            # a kill keeps everything scored so far
        print("%-34s %-5s from_behind=%.2f breasts=%.2f vlm=%s  %.1fs"
              % (stem[:34], rec["view"], behind, breasts, rec.get("vlm"),
                 time.time() - t0))

    back = sorted(s for s, r in cache.items() if r.get("view") == "back")
    print("\nwrote %s" % CACHE)
    print("gate fires on: %d/%d" % (len(back), len(cache)))
    for s in back:
        print("  %-40s %s" % (s, cache[s].get("answer", "")))


if __name__ == "__main__":
    main()
