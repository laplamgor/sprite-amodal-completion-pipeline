> **Record, not tooling.** The experiment harnesses named below (`probe.py`, `think.py`,
> `score_hand.py`, `handrank.py`, `normalize.py`, `leverage.py`) were removed once the
> axis work closed; the pipeline does not import or need them. What remains here is the
> argument for the axes as shipped, plus the two label files those numbers came from:
> `hand.json` (105 hand ranks over `build` / `waist_taper` / `hip_bias`) and `truth.json`
> (35 bust tiers). Re-measuring anything means re-writing ~20 lines of scorer against
> those labels, not restoring the old harnesses.

# Chest-size elicitation: what was tested

Metric throughout: **pairwise concordance** against `_eval/truth.json` (the
hand-ranked tier list). For every pair of images in different tiers, does the
method put the larger-busted one first? 0.5 = coin flip. It is invariant to any
monotone rescaling, so it compares methods fairly.

Reproduce with `python _eval/score.py _eval/results/<file>.json`.

## 1. Absolute questions on the VLM (Qwen2.5-VL-3B), 512 px, full figure

Six ways of asking, each 3 shuffled option orders, letter softmax.

| id | question | levels | conc |
|---|---|---|---|
| qa | "How much chest volume is under the clothing?" (the original) | 5 | 0.613 |
| qb | "Judge the bust size as it is drawn… ignore outfit, pose, art style" | 7 | 0.605 |
| qc | occlusion-aware variant of qb | 7 | 0.621 |
| qd | "what cup size is this character drawn with?" AA→G | 8 | 0.613 |
| qe | "how far does the bust project forward from the ribcage?" | 7 | 0.575 |
| qf | "rate 1-7, 1 = completely flat, 7 = largest in anime art" | 7 | **0.690** |

Wording moved the result by ~0.11. It is not the bottleneck.

## 2. Crop and resolution

| config | conc |
|---|---|
| qc, 512, full figure | 0.621 |
| qc, 512, torso crop | 0.669 |
| qc, 512, chest crop | **0.706** |

Cropping helped more than any rewording. That is the first sign that the model
is not seeing the chest, not misunderstanding the question.

## 3. Generation instead of softmax

The letter softmax is prior-dominated: for every image the top letter gets
~0.85 of the mass, and single-token completion puts ~0.85 on the word "medium"
for all 35 images. Free generation is not flat — asked to describe the
character, the model said "large" for 0639 (tier 0) and "small" for 0479
(tier 5), i.e. it can see the difference.

| id | prompt | conc |
|---|---|---|
| g1 | one word from flat/small/medium/large/huge | 0.500 (constant answer) |
| g2 | relative to "a typical C-cup anime heroine" | 0.527 |
| g3 | describe, then `BUST: <level>` | 0.666 |
| g4 | "ignore the clothing completely" | 0.500 (constant answer) |

## 4. Pairwise tournament

All 595 pairs, both orders, averaged. This is the strongest thing you can ask a
VLM to do, and it failed in an informative way:

| tier gap | accuracy |
|---|---|
| 1 | 0.48 |
| 2 | 0.34 |
| 3 | 0.32 |
| 4 | 0.23 |
| 5 | 0.08 |

Accuracy *falls* as the size difference grows. The model is not ranking bust
size at all — it is ranking how much volume the outfit occupies. Loose clothing
reads as "large", which is why 0722 (white dress, chest hidden) beat 0639
(visibly huge).

Overall conc 0.633.

## 5. Purpose-trained detector instead

`Anzhc/Anzhcs_YOLOs` → *Anzhcs Breast size det cls v8 640 y11m.pt* (YOLO11m,
15 classes, ~16k annotated illustrations, sizes relative to body proportion
rather than to the scene). Class 0 = flat … 6 = gigantic.

| config | conc |
|---|---|
| single full view, 640, conf 0.05 | 0.737 |
| full 960, conf 0.20 | 0.900 |
| full 960 + torso 640 averaged, conf 0.20 | **0.910** |

Held out: 5-fold CV, threshold tuned on 4 folds each time → **0.905** mean.
conf 0.20 was selected in every fold, so the setting is not a fluke of this set.

The two-view average is what fixes the images whose chest is partly covered by
an arm, a prop or loose clothing — 0297 (arm across the chest) and 0878 (nun
with objects on the floor) are the ones it recovers.

## 6. Bands vs tiers

Bands are the midpoints between the detector's class anchors, so a band change
means the detector moved a class rather than wobbling by a percent.

| tier | bands seen (0 = flat … 5 = huge) |
|---|---|
| 0 cannot be larger | 4, 4, 5, 5 |
| 1 very large | 4, 4 |
| 2 average (C-D) | 1, 1, 2, 3, 3, 3, 3 |
| 3 small but not flat | 0, 1, 1, 1, 1, 1, 2, 2 |
| 4 flat | 0, 0, 0, 0, 0, 1, 1, 1 |
| 5 not visible | 0, 0, 0, 1, 1, 3 |

Exact band match 21/35 (60 %), within ±1 band 32/35 (91 %).

Known disagreements, all where the artwork hides the chest:

- `1010_7156` — you put it in "not visible"; the detector reads class 6
  (*gigantic*) at conf 0.48 from one view only, and nothing from the torso view.
  The only gross outlier.
- `0443_3043`, `0394_4358`, `0414_4637` — you say flat, detector says class 1
  (*small*) at 0.71 / 0.58 / 0.29. Its class 0-vs-1 boundary is the weakest part
  of the model, which the author says himself.
- `0090_1693`, `0707_2737` — you say average, detector says *small*.

## 7. The `build` axis: reword vs bigger model vs measure

`build` scores 0.550 against the hand rank (`_eval/hand.json`) with the
pipeline's current question, and its middle two-thirds is inverted (bins 0-2
sit at 0.15-0.21 while bin 3 jumps to 0.51). Three fixes were tested on the
same 35 images, scored over the 391 cross-bin pairs.

| approach | best | delta |
|---|---|---|
| current question, `ba` | 0.529 | -- |
| **(a) 3B, `bc` + `bd` averaged** | **0.696** | **+0.167** |
| (a) 3B, `bc` alone ("thinner or heavier than a typical heroine") | 0.657 | +0.128 |
| (a) 3B, `bd` alone ("upper arm thickness vs head width") | 0.655 | +0.126 |
| (b) Qwen3-VL-4B, `bd` alone | 0.639 | +0.110 |
| (c) geometry, waist width / height | 0.616 | +0.087 |
| (b) Qwen3-VL-4B, mean of all four questions | 0.536 | +0.007 |
| (b) Qwen3-VL-4B on the current question | 0.522 | -0.007 |

### The 4B is worse, and the mechanism is visible

On the current question the 4B returns the *same* answer for 27 of 35 images
(stdev 0.014, against 0.104 for the 3B). It does not reason better; it
collapses harder onto the prior. On the most concrete question (`bd`) the two
models are level (0.639 vs 0.655), so the concreteness of the question matters
more than model size.

### Combining beats any single question

`bc` and `bd` together reach 0.696 — above either alone, and above the best
geometry. Adding the geometric waist/height on top *lowers* it to 0.639, so
the silhouette feature is redundant against the two questions. Same principle
as averaging two views for the bust detector.

### Geometry on its own

| axis | best feature | geo | VLM |
|---|---|---|---|
| `build` | waist width / height | 0.616 | 0.549 |
| `waist_taper` | waist width / head width | 0.607 | 0.716 |
| `hip_bias` | hip width / height | 0.707 | 0.540 |

Geometry beats the VLM on `build` and `hip_bias` but loses badly on
`waist_taper`, so it is not a drop-in replacement for the VLM pass. Features
are written to `input/shape/<stem>.json` by `shape_measure.py`; only 19/35
images have separated legs, so hip and thigh landmarks are unreliable on the
rest.

### Caveats

- `bc + bd` was chosen after seeing these results, so 0.696 is optimistic;
  expect ~0.65-0.70 held out.
- 391 pairs from 35 images: differences under ~0.05 are inside the noise. Only
  "the current question is bad" and "`bc`+`bd` is best" are clear.
- Still far below the bust detector's 0.91. `build` may simply not be a thing
  these models can see; consider a purpose-trained regressor if it matters.

### Shipped

`vlm_axes.py` now asks `bc` and `bd` as separate axes and stores their mean as
`build`. Measured on the shipped code path the mean scores **0.737** (vs 0.547
for the old question on the same corpus), each question alone scores 0.693 /
0.685, and the probe's independent shuffle draw scores 0.675 for the same pair.

The gap against the probe's 0.696 is *not* a methodology difference: both
harnesses average three shuffled option orders with identical preprocessing.
They simply drew different permutations (`random.Random(q-o)` per order in
`probe.py`, `random.Random(axis)` once in `vlm_axes.py`). Two draws of the same
measurement, so treat the honest estimate as **~0.70 ± 0.04**, not 0.737.

`rederive.py` ranks the combined score across the corpus before use, so the
fixed frame cuts keep their meaning; `limb` and `age_imp` were removed.

Where the signal actually is, on the shipped `build`:

| subset | conc | pairs |
|---|---|---|
| all four bins | 0.737 | 391 |
| bin 3 vs everything else | 0.927 | 96 |
| bins 1-2 (the bulk) | 0.671 | 208 |
| bins 0-1 | 0.590 | 39 |

So the heavy tail is nearly solved and the ordering *within* the ordinary
two-thirds is real (0.671) rather than inverted, which is what makes a
rank-based level map safe: the middle is compressed in raw score but not
random. The old single question scored 0.431 within bins 0-2 - worse than a
coin flip.

## 8. Thinking mode: 2B-Thinking vs 4B and 3B

### The harnesses never decoded

Both `probe.py` and `vlm_axes.py` do one forward pass and softmax over the
answer-letter token ids. There is no decoding step, so no reasoning trace can
exist in that readout even with Thinking weights loaded. The 4B's collapse to
one answer on 27/35 images was measured under those conditions.

`_eval/think.py` does the other thing: generate with `Qwen3-VL-2B-Thinking`
(whose chat template hard-codes the think prefix - there is no switch), strip
the block, parse the letter. Same questions, same images, same 3 shuffles, same
scoring; only the readout differs.

| axis | 3B (shipped) | 4B | 2B thinking |
|---|---|---|---|
| `build` (pair mean) | **0.737** | 0.501 | 0.441 |
| `waist_taper` | **0.714** | 0.640 | 0.524 |
| `hip_bias` | 0.540 | **0.707** | 0.567 |

Each model's own shuffle draw differs, so compare within a row only to ~0.04.

### Thinking made 2B decisive, not accurate

| question | most common answer | share |
|---|---|---|
| `bh` hip vs shoulder | "wide hips, narrow shoulders" (top option) | 31/35 |
| `bw` waist taper | "slight taper" | 27/35 |
| `bc` thinner/heavier | one value | 11/35 |

The high spread (stdev 0.20-0.23) is misleading: it is not sensitivity to the
image, it is a bimodal "confidently one of two options". `bh` is the clearest
case - 31 of 35 land on the maximum, the 5 images ranked widest are inside that
pile, and the axis separates almost nothing. It still scores 0.567 only because
the 4 images it does separate happen to fall on the right side.

Reasoning length averaged 188 (`bc`), 325 (`bd`), 176 (`bw`) and 51 (`bh`)
words; 5 of 420 answers were unparsed at the 512-token cap.

### Cost

18.6 min for 35 images x 12 rows = **32 s/image**, i.e. ~53 min per 100 images
against 2.7 s/image (~5 min) for the 3B 13-axis pass - about **12x slower** for
a quarter of the axes. VRAM was 8.4 GB during the run versus 8.0 GB peak for
the 3B pass: the smaller weights are cancelled by the KV cache for 12 long
sequences.

### Methods correction (moves some section 7 numbers)

The first scorer used here read

    hit += 1 if (d > 0) == (bins[a] > bins[b]) else (0.5 if d == 0 else 0)

which never reaches the tie branch: when `d == 0` it evaluates
`False == (bins[a] > bins[b])`, so a tie counts as a full hit whenever the first
image sits in the lower bin and a full miss otherwise. On coarse axes this is
fatal - `hip_bias` has 3 options, so 3 shuffles give only 1/6 steps and 130 of
150 pairs were ties. The thinking 2B was briefly scored at 0.900 on `hip_bias`
instead of 0.567.

`_eval/score_hand.py` is now the single scorer, with ties worth 0.5;
`handrank.py` and `normalize.py` delegate to it. Section 7's numbers were
re-derived with it: continuous and 5-option axes moved by <=0.02,
`waist_taper` 0.718 -> 0.714, `build` 0.739 -> 0.737, and no conclusion
changed.

## 9. Chest wording, properly isolated, and the flat-chest LoRA

### Token bleeding is false here -- silence is worse

One image (0381, source class 0.00), four prompts, everything else identical:

| prompt | rendered class |
|---|---|
| no chest words at all (`soft flanks, straight waist, visible ribs`) | **1.90** |
| shipped fragment + booru tags | 1.14 |
| fragment, tags removed | 1.09 |
| single tag only (`flat chest, ...`) | 1.14 |

So "breasts"/"chest" tokens do not inflate the render on this checkpoint; they
are the only thing holding it back. The words buy 0.75 of the 1.9 prior and
**saturate at one phrase** -- a bare `flat chest` is as good as a three-phrase
clause. That also explains every earlier "wording does nothing" result: there
was nothing above the saturation point to find.

### The LoRA

`rzgar/Eva-Qwen-Flat-Chest` (`Eva_Flat_Chest.safetensors`, 225 MB), trained on
Qwen-Image rather than Edit, so cross-base. Applied only below 0.26 bust,
strength 1.0, trigger word before the other chest phrases.

| | n | mean bias vs source |
|---|---|---|
| before | 20 | **+1.31** |
| after | 20 | **+0.52** |

15 improved, 5 unchanged, 0 worse. Largest: 0722 3.69 -> 1.16, 0707 2.50 -> 1.00,
0018 2.00 -> 0.78, 0329 2.00 -> 1.00, 0921 1.82 -> 1.00.

Three things worth remembering:

- **Strength 0.7 does nothing at all** (0.93 / 1.00 / 1.00, identical to no
  LoRA). The dose response is a cliff, not a slope; 1.0 is the usable setting.
- **Trigger position matters.** With `Eva_Flat_Chest` appended after the tags,
  0381 rendered 0.91; before them, 0.30. Same LoRA, same strength.
- **Cost is ~0**: 74.3 s/image over 18 images against 73.5 s/image without it.
  The isolated warm pair suggested +3.3 s/image for the one-time patch, which
  disappears into chunk-level noise.

Still unresolved: five images sit at exactly class 1.00 and do not move (0248,
0653, 0547, 0414, 0231). Detection on 0248 did not change but the user judged
it visibly better, so the detector is saturating below a class boundary there.
