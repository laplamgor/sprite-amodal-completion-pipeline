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

## 10. Where the detection stage's second are (ADR-0012)

Five SAM3 probes, one image (0528_1165), chunk 1, 4 steps, timers inside the node
(AMODAL_TIMING=1):

| quantity | refine 2 | refine 1 | refine 0 |
|---|---|---|---|
| backbone, once | 0.34-0.38 s | 0.28-0.29 s | 0.29 s |
| detect incl. refine | 8.9-9.5 s | 5.9-6.9 s | 2.2 s |
| refine, 27 objects | 7.08-7.15 s | 3.60-4.06 s | 0.00 s |

Refinement is 72% of detection, at 0.13 s per object-iteration. The five-prompt
head without it is 2.2 s and one image encoding is ~0.35 s.

Sharing the backbone across the five prompts therefore saves 4 x 0.35 = 1.4 s.
Hoisting the tracker trunk out of the refine loop saves 3.1 s. Together ~4 s/image,
~8% of the 47 s budget, with dumped masks bit-identical on two images at two
thresholds (0/1 336 320 px and 0/1 228 800 px differ, both the edited union and
the protection union).

### Wall clock is not a usable instrument below ~25% here

Identical 1-image runs measured 108, 108, 96, 90, 48, 34, 40 s in execution order
across one session. Backbone timing is bimodal: 3.0-3.2 s on the first call after
idle, 0.29-0.38 s warm. Refine timing reproduced to +-0.1 s.

This retracts two earlier readings as artefacts: --sam-refine 1 at 148 vs 144 s
("no gain", actually halves refine: 7.1 -> 3.8 s) and --probe-merge at 136 vs
122 s ("slower", actually faster once warm).

### The tail is not CPU-bound (ADR-0012 follow-up)

AmodalFinalize, the only CPU-side node in the tail, timed at **0.160 s total**
(render resize 0.052, mask resize 0.017, math+blur 0.057, edited-mask 0.023,
writeback 0.009) on 0528_1165 at 1392x960.

So porting the tail to torch would target ~1.5% of an ~11 s stage. Rejected
without further work: the remaining time is VAEDecode and BiRefNet, both already
on the GPU. Note both run at low utilisation there (13-81% across the tail), so
the cost looks like launch/preprocessing overhead rather than compute -- but that
is a different question and is not yet measured.

## 11. Chunking only works with stage ordering (ADR-0013)

At --chunk 7 ComfyUI requested every model **seven times** in one prompt: it
schedules depth-first, so each image is rendered end to end before the next
starts. Per image that is ~24 s of text-encoder load and ~14 s of unet load on a
card where the two (8.74 + 9.36 GB) cannot co-reside.

Three identity gates (AmodalStageGateMask, AmodalStageGateLatent) force
detect-all, encode-all, sample-all. Same three images, cold, chunk 3:

| arm | total | per image |
|---|---|---|
| depth-first | 272 s | 90.7 s |
| stage-ordered (3 gates) | 182 s | 60.7 s |

Loads per 3-image chunk: SAM3Clip 3->1, SAM3 3->1, text encoder 3->1, ControlNet
3->1, BiRefNet 3->1. The unet stays at 3 because its ControlNet patch is
built per image from that image's mask.

Masks compared pixel-wise on all three images, edited and protection unions:
0 px differ.

--reserve-vram 0.2 was tested in the same way and rejected: 279 s vs 272 s.

Correction to section 10 and to the README: "bigger chunks amortise reloads" was
false before this change, and the earlier ~68 min / 100 images estimate was
~138 min.

## 12. Back-facing sources: prompt routing, and the 4-step denoise quantum

0479 has the torso away from the camera (head over the shoulder, long sleeves).
The shipped front prompt rebuilds it as a front with both arms swapped. Five
levers, one image, same seed / 4 steps / chunk 1:

| render | low-bust branch | prompt | denoise | result |
|---|---|---|---|---|
| baseline | on | front | 1.0 | front torso, breasts, arms re-synthesised |
| (1) | off | front | 1.0 | still front, chest larger than baseline |
| (1)+(2) | off | back | 1.0 | back, arms in the source pose |
| (1)+(2)+(3) | off | back | 0.85 | byte-identical pixels to (1)+(2) |
| (1)+(2)+(3) | off | back | 0.80 | back, but sleeve shading survives (arm warmers) |

Why 0.85 did nothing: `KSampler.set_steps` uses `new_steps = int(steps/denoise)`
then keeps the last `steps+1` sigmas. At 4 steps, `int(4/0.85) = 4`, so the
schedule is identical to denoise 1.0. 0.80 gives `int(4/0.8) = 5` -- the first
setting that changes anything. This also re-reads ADR-0008's "denoise 0.90 =
0.00": that arm never changed the schedule.

View detection: the 3B VLM is unusable here. Letter options gave p(back) 0.10-0.52
on 0479 over five phrasings; teacher-forced continuations 0.09; greedy answers
called 8/20 images "back"; the directive phrasing called 20/20 "back". WD SwinV2
v3 tags separate it: `from_behind` 0.08 / `breasts` 0.02 on 0479, `from_behind`
<= 0.02 or `breasts` >= 0.21 on the other 19. 0878 is the reason both terms are
required (from_behind 0.26, breasts 0.92, front-facing crawl).

Gate = `from_behind >= 0.05 and breasts < 0.20`, written to
`output/axes/_views.json`.

The labels took three passes. The first two were mine and wrong: 0313 was read
as a front (flat sailor back panel read as a chest) and 0878 as a front (a crawl
where the chest is visible from the side). The operator corrected both. Ground
truth for the whole thread:

| stem | truth | from_behind | breasts | 4B-Instruct | 4B-Thinking |
|---|---|---|---|---|---|
| 0479 | back | 0.083 | 0.021 | back | back |
| 0878 | back (3/4) | 0.263 | 0.923 | back | back |
| 0958 | back (3/4) | 0.019 | 0.072 | back | back |
| 0313 | back | 0.488 | 0.007 | **chest** | **chest** |
| 0653 | front (hips turned) | 0.002 | 0.029 | **back** | **back** |

Every other front in the 22 scores `from_behind` <= 0.003.

| instrument | backs found (of 4) | false backs (of 18 fronts) |
|---|---|---|
| WD tagger, `from_behind >= 0.01` | **4** | **0** |
| Qwen3-VL-4B | 3 (misses 0313) | 1 (0653) |
| tagger and 4B | 3 (misses 0313) | 0 |

The floor sits in a 0.002-0.019 gap; it separates this corpus perfectly and will
not survive the next one. The `breasts < 0.20` veto was fitted to the wrong 0878
label and is removed.

Thinking mode answers identically at ~5x the tokens; two of twelve runs never
finished inside 600 tokens. Checkpoint deleted.

### The prompt-only alternative

One sentence naming both sides ("seen from behind her bare back faces the camera,
seen from the front her bare chest faces the camera"), no label at all:

| stem | truth | neutral prompt |
|---|---|---|
| 0479 | back | back |
| 0313 | back | back |
| 0878 | back | back |
| 0958 | back | **front** |
| 0002 | front | front |
| 0942 | front | front |

So the prompt fixes what the gate misses (0313) and fails what the gate gets
(0958). Neither is complete; the gate is shipped because a false "back" is the
unrecoverable direction and it has none, with `--assume-back` for the rest.

### Expanded back-view sample (27 images, 8 backs)

Four more backs and one mixed pose were added by the operator: 0228, 0450, 0584,
0941 (backs) and 0390 (upper body back/side, hips front). `from_behind`:

| stem | from_behind | 4B | truth |
|---|---|---|---|
| 0228 | 0.597 | back | back |
| 0313 | 0.488 | chest | back |
| 0941 | 0.462 | chest | back |
| 0584 | 0.436 | back | back |
| 0450 | 0.374 | chest | back |
| 0878 | 0.263 | back | back |
| 0479 | 0.083 | back | back |
| 0958 | 0.019 | back | back |
| 0390 | 0.011 | front | mixed |
| other 18 fronts | <= 0.003 | chest | front |

Tagger 8/8, no false backs. 4B 5/8 (the three misses are all 3/4 backs with the
chest partly visible -- the question "chest or back" has the wrong shape for
them). Gate simplified to the tagger alone; `--vlm` still records the answer.

All five new images measure `bust` 0.05 (lowest class) because no chest is
visible -- without the gate they would also take the flat-chest LoRA and
"pettanko, tiny breasts, flat chest", pushing them further toward a front.


### Bust wording on back views: describe what was measured

A back view hides the chest, so the detector reports the lowest class: all five
new backs measure 0.05, and six of the eight backs sit at 0.05-0.30.

One round tried dropping the chest clause on backs below bust 0.5, so the prompt
would stop asserting "androgynous flat chest, smooth flat chest plane, no
underbust crease". The operator's rule is the opposite: keep describing what was
measured, including flat/small -- only the assertive low-bust branch (LoRA plus
"pettanko, tiny breasts, flat chest") stays off on backs.

| back view | bust | chest clause |
|---|---|---|
| 0228, 0584, 0958 | 0.05 | kept |
| 0313, 0390, 0479 | 0.22 | kept |
| 0450 | 0.30 | kept (medium) |
| 0941 | 0.52 | kept |
| 0878 | 0.72 | kept |

A/B on 0228, same seed: without the clause the spine and ribcage contour get
drawn; with it the back is a smooth flat plane. Final rule keeps the clause.
`BACK_BUST_MIN` (default 0.0 = never drop) keeps the subtractive path available.

## 13. Garment-sized protection masks, and the LoRA gate at 0.50 (2026-11-05)

Operator report, in priority order: three images were "not nude at all" (0932
black latex suit, 0547 white hooded cloak, 0722 white wedding dress); four had
"breast inpainted as too large and then cropped by mask" (0479, 0497, 0528,
0687); two back views got a side-boob (0313, 0479); and three loose-garment
images re-posed an arm (0921, 0941, 0958).

### The equipment protection mask was the garment

`--dump-mask` on the three dressed images put the *edited* mask at 2.2 / 4.0 /
3.8 % of the canvas. The inpaint was running; it had almost nothing to run on.

Per-prompt areas, one prompt per SAM3 pass (`_eval/probe_masks.py`, threshold
0.25, % of canvas):

| stem | outfit | garments | equipment | edited mask |
|---|---|---|---|---|
| 0932 | 2.2 | 12.8 | **22.6** | 2.2 |
| 0722 | 25.3 | 34.1 | **34.1** | 4.0 |
| 0547 | 23.9 | 27.1 | **27.9** | 3.8 |

Protection is subtracted from clothing, so the equipment mask alone erased the
garment. Per word (`_eval/probe_words.py`):

| word | 0722 dress | 0547 cloak | 0932 suit | worst real prop |
|---|---|---|---|---|
| weapon | 26.3 | 0.2 | 0.1 | 1.5 |
| gun | 32.8 | 0.6 | 0.1 | 1.5 |
| shield | 25.4 | 0.3 | 17.1 | 0.2 |
| backpack | 22.6 | 23.4 | 9.4 | 8.3 |
| bag | 3.5 | 22.5 | 1.3 | 7.2 |
| machinery | 33.8 | 27.2 | 20.8 | 2.0 |
| equipment | 1.3 | 0.7 | 22.6 | 2.0 |

Two levers were measured and rejected before the cap:

| lever | edited mask (0932 / 0722 / 0547) | verdict |
|---|---|---|
| 25-term garment list | 3.2 / 4.0 / 4.0 % | no effect |
| `--parse` (SegFormer clothes) | 2.4 / 4.1 / 4.1 % | no effect |

`--parse` is worth remembering: the parse mask itself was 10.2 / 32.2 / 28.5 %
and the final mask did not move, because the same protection union was
subtracting it. That is what pointed at protection instead of detection.

The shipped fix is the size cap (ADR-0012): a single protection input above
`PROTECT_MAX_AREA = 15` % is dropped. Final masks 18.2 / 36.9 / 33.2 %, all
three nude. Controls 0313 (rig, guns, armour, thigh gear) and 0958 (lantern,
fox mask, ribbons) are visually unchanged.

Trap: `max_area` must be an *optional* input on `AmodalMaskUnionFill`. As a
required one the server rejected every prompt that did not send it -- the
clothing union and `make_ui_workflow` -- with `required_input_missing`.

Accepted cost: with the mask now covering the whole dress/cloak, the model
re-poses the arms inside that large area, and 0722's bouquet is painted away.
Previously those images were fully dressed, so this is the better failure.

### The LoRA gate moved from 0.26 to 0.50 and now runs on backs

The four "cropped" images are one fault: below average the default render paints
a chest larger than the source, wider than the clothing mask, and the part over
the arm is discarded at the mask edge, leaving a straight vertical cut.

| image | source | default | with LoRA |
|---|---|---|---|
| 0578 | 0.35 | 0.52 | 0.38 |
| 0528 | 0.31 | 0.47 | 0.28 |
| 0687 | 0.38 | 0.38* | 0.26 |
| 0902 | 0.47 | 0.66 | 0.41 |
| 0497 | 0.47 | 0.55 | 0.27 |
| 0450 | 0.30 | 0.42 | 0.22 |
| 0878 (control, above gate) | 0.72 | 0.63 | -- |

\* 0687's class is unchanged but its default render is the clearest crop: the
bust's outer edge is a vertical line at the arm; the LoRA version fits inside
the mask.

On backs the source measurement is the only signal -- the detector reads 0.05 on
both 0313 and 0479 *renders*, because a back hides the chest. Both sources are
0.22, both painted a full side-boob; the LoRA removes it. The tags stay at 0.26
and stay off on backs (`--back-low-tags` opts in): they assert a front chest.
`--no-back-low` now skips LoRA and tags together.

### Rejected in the same round

| lever | image | result |
|---|---|---|
| `--grow 32` | 0528 | painted chest got larger and merged into the arm |
| `--grow 8` | 0921 | arm/hand rebuilt the same way as the default |
| `--parse` | the three dressed | final mask unchanged (see above) |

The loose-garment arm/third-arm problem (0921, 0941, 0958) is *not* fixed by
mask growth in either direction; it is the model re-posing inside a large edit
area. 0958's low IoU (0.708) is the far hand rebuilt behind the hip and a short
rear foot, not a mask bug.
### What the 27-image run actually produced

Whole corpus, `evaluate.py --no-sheets`: mean IoU 0.869, median 0.916, mean
dAlpha -12.7 % (before this round: 0.916 / 0.956 / -8.0 %). The drop is
concentrated in the three images that are now nude -- 0547 0.499, 0722 0.498,
0932 0.725. Removing a dress removes silhouette, so -50 % alpha there is the
goal, not a regression. Every other image is unchanged or within a few points.

Bust, source -> render over all 27, mean delta +0.023. The four crop cases:
0497 0.47 -> 0.27, 0528 0.31 -> 0.22, 0687 0.38 -> 0.22, 0450 0.30 -> 0.26.
Nothing on the front side is flatter than its source by more than one class.
The two remaining outliers are backs -- 0479 +0.16, 0584 +0.33 -- where the
detector cannot see the chest; the visual A/B shows a flat back on both.

### Trap: the back gate was inverted

The per-image condition read `not (back and not a.no_back_low)`: backs skip the
low-bust branch *unless* the flag is passed. The first 27-image run therefore
applied the LoRA to 15 fronts and to no back at all, and nothing failed loudly.
Caught by pixel-diffing the `_blo` A/B against the default -- 0 px differ with
`--no-back-low` absent, which is impossible if the LoRA had run. Fixed to
`not (back and a.no_back_low)`.

The pre-flight count printed "15 low-bust image(s)" for the same reason: it
excluded backs from the LoRA estimate even though the gate was meant to include
them. It now uses the same condition as the per-image gate (22 images below
0.50, 15 fronts + 7 backs). The grouping/sort used that list too.

Do not use the detector to rank backs: 0479's pre-LoRA render read 0.05 and its
post-LoRA render 0.38 with no visible chest either time.

