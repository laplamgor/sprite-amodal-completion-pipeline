"""Render the calibrated body prompt from cached VLM axes.

This is the single renderer the batch uses. Layout of the fragment:

    <chest>  always emitted, so breast size can never fall back to the model prior
    <frame>  build / proportion, only when the axis is low
    <anatomy> up to 4 terms, one per family, at most 2 "absence" statements
    <style>  up to 2 terms

Applied to the whole folder after vlm_axes.py:

    python rederive.py                 # fit calibration if absent
    python rederive.py --refit         # refit on the current corpus
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402
import vlm_axes as V

AX = C.AXES
CAL = os.path.join(AX, "_calibration.json")

FAMILY = {
    "visible ribs": "ribs", "thoracic arch": "ribs", "ribcage contour": "ribs",
    "ab definition": "abdomen", "flat abdomen": "abdomen", "soft belly": "abdomen",
    "linea alba": "abdomen",
    "obliques": "flanks", "serratus": "flanks",
    "inframammary fold": "underbust", "underbust shadow": "underbust",
    "hip bones": "hips", "thigh gap": "hips",
    "waist definition": "waist",
    "collarbone": "collarbone", "navel depth": "navel", "skin tension": "skin",
}

# low pole is a real statement for these; for the rest, "absence" is not written
LOW_OK = {
    "visible ribs", "thoracic arch", "ribcage contour", "ab definition",
    "obliques", "soft belly", "hip bones", "collarbone",
    "inframammary fold", "underbust shadow", "navel depth", "waist definition",
    "thigh gap", "skin tension",
}

# terms that only make sense on a fuller chest
CHEST_DEPENDENT = {"inframammary fold", "underbust shadow"}

# ---------------------------------------------------------------------------
# Chest phrasing.
#
# `bust` now comes from bust_measure.py (a purpose-trained detector), not from
# the VLM, and is on a 0..1 scale where 0.5 is the anime "average" of C/D.
# Bands are the midpoints between the detector's class anchors
# (0.05 / 0.22 / 0.38 / 0.55 / 0.72 / 0.88), so a band change means the
# detector actually moved a class, not that it wobbled by a percent.
#
# Several phrasings per band: they are all saying the same size, but they give
# different emphasis (size word, underbust crease, cleavage, hang). Pick one per
# image with --chest-variant, or leave it to the per-image seed so the same file
# always renders the same phrase.
BANDS = [  # (upper bound, name)
    (0.10, "flat"), (0.26, "small"), (0.42, "medium"),
    (0.59, "average"), (0.78, "large"), (9.99, "huge"),
]

CHEST_BANK = {
    # The flat and small registers avoid size-positive words ("round",
    # "cleavage", "firm", "full") because nothing above the saturation point
    # helps: a controlled same-image test showed every flat wording lands at
    # class 1.09-1.14 while no chest words at all lands at 1.90 (ADR-0008).
    # The older claim that one phrasing beat another was confounded by which
    # images got which phrasing -- do not re-tune these words.
    "flat": [
        ["completely flat chest", "no bust volume", "flat sternum"],
        ["androgynous flat chest", "smooth flat chest plane", "no underbust crease"],
        ["flat chest wall", "no chest curve", "flat pectoral plane"],
        ["very flat chest", "flat chest plane", "no bust at all"],
    ],
    "small": [
        ["very small breasts", "barely a curve", "no cleavage"],
        ["small bust", "slight outward curve only", "no underbust crease"],
        ["petite low-profile bust", "minimal chest curve", "small areolae"],
        ["A-cup at most", "small flat chest", "shallow underbust"],
    ],
    "medium": [
        ["medium breasts", "soft rounded bust", "natural hang"],
        ["medium breasts", "soft underbust crease", "moderate bust"],
        ["medium round breasts", "moderate cleavage", "full soft bust"],
        ["average bust", "gentle full curve", "soft breast shape"],
    ],
    "average": [
        ["average anime bust", "full round breasts", "clear cleavage"],
        ["full breasts", "rounded bust", "visible underbust crease"],
        ["C-cup breasts", "full round bust", "soft cleavage line"],
        ["full soft breasts", "natural hang", "clear cleavage"],
    ],
    "large": [
        ["large breasts", "heavy full bust", "deep cleavage"],
        ["large round breasts", "pronounced underbust crease", "full bust"],
        ["very full breasts", "heavy bust", "deep cleavage"],
        ["large soft breasts", "visible underboob", "heavy bust"],
    ],
    "huge": [
        ["huge breasts", "very heavy bust", "deep cleavage"],
        ["extremely large breasts", "heavy overhanging bust", "deep crease"],
        ["massive soft breasts", "deep cleavage", "heavy underboob"],
        ["huge round breasts", "pronounced underboob shadow", "heavy bust"],
    ],
}


def band_of(bust):
    for i, (hi, name) in enumerate(BANDS):
        if bust < hi:
            return i, name
    return len(BANDS) - 1, BANDS[-1][1]


def chest_terms(bust, stem="", variant=None):
    """Chest clause for a measured bust value.

    variant=None -> chosen from the stem, so the same image is reproducible but
    different images are not worded identically. variant=N forces one phrasing
    across the whole run, which is how you A/B them.
    """
    idx, name = band_of(bust)
    bank = CHEST_BANK[name]
    if variant is None:
        h = 0
        for ch in stem:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        k = h % len(bank)
    else:
        k = variant % len(bank)
    out = list(bank[k])
    if idx <= 1:                      # flat / small: assert a flat torso too,
        out.append("flat stomach")    # the rounded-belly failure mode
    return out


# lean-body terms that contradict a heavy bust; suppressed when the chest is full
SUPPRESS_WHEN_FULL = {
    "visible ribs", "ribcage contour", "ab definition", "linea alba",
    "serratus", "obliques", "skin tension",
}
FULL = 0.59          # "large" and above

# Terms whose high pole asserts muscle *definition*.  The muscle axis is
# saturated low on a corpus of soft characters -- across 35 images it never
# leaves 0.07..0.355, where "slight tone" is 0.33 -- so the p10-p90 stretch
# turned that noise into "visible serratus" (9/35), "visible linea alba" (5/35),
# "chiselled obliques" (3/35) and "defined abs" (1/35) on characters the model
# itself calls untoned.  Same pathology as age_imp, so same treatment: assert
# them only when the raw axis is clearly past "slight tone".  Low poles are
# still free to fire, which is what a soft corpus actually supports.
MUSCLE_ASSERT = {"ab definition", "obliques", "serratus", "linea alba",
                 "skin tension"}
MUSCLE_MIN = 0.45


def frame_terms(axes, chest_len=0):
    """Frame words, read from the percentile-mapped `build` (see main()).

    The axis is mapped to a rank across the corpus first, so 0.5 is the median
    image and these cuts mean what they say: 0.24 is the bottom ~13%, 0.34 the
    bottom ~27%.  Reading the raw axis here was the bug -- the raw score piles
    most of the corpus into one narrow band, so the cuts fired on the wrong
    images (and on none of the three that most needed it).
    """
    out = []
    build = axes.get("build", 0.5)
    prop = axes.get("proportion", 0.70)
    if build < 0.24:
        out.append("very slender frame")
    elif build < 0.34:
        out.append("slender frame")
    if prop < 0.35:
        out.append("petite proportions")
    return out[:1 if chest_len >= 3 else 2]


def render(scaled, axes, stem="", variant=None, cap=3):
    bust = axes.get("bust", 0.5)
    chest = chest_terms(bust, stem, variant)
    parts = chest + frame_terms(axes, len(chest))

    flat = bust < 0.26
    items = []
    for n, v in scaled.items():
        dev = abs(v - 0.5)
        if dev < 0.12:
            continue
        if v < 0.5 and n not in LOW_OK:
            continue
        if flat and n in CHEST_DEPENDENT:
            continue
        items.append((n, v, dev))

    best = {}
    for n, v, dev in items:
        if bust >= FULL and n in SUPPRESS_WHEN_FULL:
            continue
        fam = FAMILY.get(n, n)
        if fam not in best or dev > best[fam][2]:
            best[fam] = (n, v, dev)
    items = list(best.values())

    # a soft corpus must not be talked into abs by the percentile stretch
    if axes.get("muscle", 0.0) < MUSCLE_MIN:
        items = [t for t in items
                 if not (t[1] > 0.5 and t[0] in MUSCLE_ASSERT)]

    pos = [t for t in items if t[1] > 0.5]
    neg = sorted([t for t in items if t[1] < 0.5], key=lambda t: -t[2])[:2]
    items = sorted(pos + neg, key=lambda t: -t[2])[:cap]

    phrase = {n: (hi, lo) for n, _, hi, lo in V.DERIVED}
    for n, v, _ in items:
        hi, lo = phrase[n]
        if v > 0.5:
            parts.append(hi)
        elif lo:
            parts.append(lo)

    seen = set()
    uniq = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def style_terms(axes, cap=2):
    out = []
    for aid, lo_txt, hi_txt in V.STYLE:
        v = axes.get(aid, 0.5)
        if abs(v - 0.5) >= 0.12:
            out.append((abs(v - 0.5), hi_txt if v > 0.5 else lo_txt))
    out.sort(key=lambda t: -t[0])
    return [t[1] for t in out[:cap]]


def percentile_rank(values):
    """Average-rank percentile in 0..1; ties share one rank."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    out = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            out[items[k][0]] = r / (n - 1) if n > 1 else 0.5
        i = j + 1
    return out


def calibrate_build(data):
    """Map the combined build score to a rank across the corpus, 0.15..0.85.

    The two-question build score ranks well but its levels are compressed: most
    of the corpus lands in one narrow band near the bottom.  Feeding that to
    `1 - build` says "visible ribs" for almost everyone, and to frame_terms says
    "very slender" for almost everyone.  A rank fixes the levels without
    touching the ordering, which is the part that was measured (0.70).

    The stretch is the same 0.15 + 0.70 * v used for the derived terms, so
    0.5 stays the middle of the corpus and the fixed cuts keep their meaning.
    """
    raw = {stem: ax.get("build", 0.5) for stem, ax in data.items()}
    pct = percentile_rank(raw)
    return {stem: 0.15 + 0.70 * p for stem, p in pct.items()}


def load_measured():
    """bust_measure.py output, if it has been run."""
    out = {}
    if not os.path.isdir(C.BUST):
        return out
    for f in glob.glob(os.path.join(C.BUST, "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                out[os.path.splitext(os.path.basename(f))[0]] = json.load(fh)["bust"]
        except (KeyError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="refit the calibration")
    ap.add_argument("--chest-variant", type=int, default=None,
                    help="force one chest phrasing (0-3) instead of per-image")
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(AX, "*.json")))
             if not f.endswith("_calibration.json")]
    data = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data[os.path.splitext(os.path.basename(f))[0]] = json.load(fh)["axes"]
    if not data:
        raise SystemExit("no cached axis files — run vlm_axes.py first")

    # the detector's measurement replaces the VLM's guess wherever we have one
    measured = load_measured()
    replaced = 0
    for stem, b in measured.items():
        if stem in data:
            data[stem]["bust"] = b
            replaced += 1
    print("bust: %d measured by detector, %d still on the VLM axis"
          % (replaced, len(data) - replaced))

    # `build` is consumed on exactly one scale from here on: the corpus rank.
    # Every formula below and frame_terms() read the same calibrated value.
    build_cal = calibrate_build(data)
    cal_axes = {stem: dict(ax, build=build_cal[stem]) for stem, ax in data.items()}

    names = [n for n, _, _, _ in V.DERIVED]
    raw = {stem: {n: fn(ax) for n, fn, _, _ in V.DERIVED}
           for stem, ax in cal_axes.items()}

    if os.path.exists(CAL) and not a.refit:
        with open(CAL, encoding="utf-8") as fh:
            cal = json.load(fh)
        print("using frozen calibration (%d terms)" % len(cal))
    else:
        cal = {}
        for n in names:
            vals = sorted(raw[s][n] for s in raw)
            lo = vals[max(0, int(len(vals) * 0.10))]
            hi = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
            if hi - lo < 1e-4:
                hi = lo + 1e-4
            cal[n] = [lo, hi]
        with open(CAL, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, indent=1)
        print("fitted calibration on %d images" % len(raw))

    scaled = {}
    for stem, r in raw.items():
        s = {}
        for n in names:
            lo, hi = cal[n]
            v = (r[n] - lo) / (hi - lo)
            s[n] = max(0.0, min(1.0, 0.15 + 0.70 * v))
        for x, y in V.PAIRS:
            if abs(s[x] - 0.5) >= abs(s[y] - 0.5):
                s[y] = 0.5
            else:
                s[x] = 0.5
        scaled[stem] = s

    print("\n%-24s %5s %5s %5s  %s" % ("stem", "bust", "build", "prop", "prompt"))
    for stem, ax in data.items():
        parts = render(scaled[stem], cal_axes[stem], stem, a.chest_variant)
        style = style_terms(ax)
        print("\n%-24s %5.2f %5.2f %5.2f" % (
            stem[:24], ax.get("bust", 0.5), build_cal[stem],
            ax.get("proportion", 0.5)))
        print("  body :", ", ".join(parts))
        print("  style:", ", ".join(style) if style else "(none)")

        path = os.path.join(AX, stem + ".json")
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["parts_cal"] = parts
        rec["style"] = style
        rec["scaled"] = scaled[stem]
        rec["build_cal"] = round(build_cal[stem], 3)
        if stem in measured:
            # axes.bust is the VLM's guess; this is what actually drove the prompt
            rec["bust_measured"] = measured[stem]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
