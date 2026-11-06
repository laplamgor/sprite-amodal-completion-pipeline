# sprite-amodal-completion-pipeline

A batch pipeline for **amodal completion** of transparent-background character
art: the clothing regions are re-synthesised as bare skin, while pose, identity,
hair, accessories and props are preserved. Output matches the source dimensions
exactly and stays RGBA with the original alpha.

One ComfyUI prompt per chunk of images. No manual masking, no per-image prompt
editing. Every parameter is derived from the individual file, never from a
remembered character.

> **What it produces.** The output is the same drawing with the garment regions
> replaced by skin. This is an image-editing research tool; inputs are expected
> to depict adults.

---

## Layout

```
ComfyUI_windows_portable_v043/
├─ sprite-amodal-completion-pipeline/
│  ├─ config.py              paths + host (single source of truth)
│  ├─ run_folder.py          batch entry point
│  ├─ vlm_axes.py            per-image body/style axes from Qwen2.5-VL
│  ├─ bust_measure.py        chest size from a purpose-trained detector
│  ├─ view_axis.py           front/back gate: booru tagger + Qwen3-VL-4B
│  ├─ rederive.py            calibrated prompt renderer
│  ├─ depth_prepass.py       DA3 depth maps (input for image_profile)
│  ├─ image_profile.py       measured per-image body profile
│  ├─ make_ui_workflow.py    regenerates the UI graph from run_folder constants
│  ├─ comfy_client.py        ComfyUI HTTP helpers (submit / wait / history)
│  ├─ evaluate.py            IoU + alpha QA, contact sheets
│  ├─ models/breast_size_y11m.pt   chest detector weights (39 MB)
│  ├─ models/wd-swinv2-tagger-v3.onnx   booru tagger weights (467 MB)
│  ├─ input/                 source PNGs, flat      (gitignored)
│  ├─ output/                <stem>_amodal.png      (gitignored)
│  │  ├─ axes/               <stem>.json   ← vlm_axes → rederive
│  │  │                       _views.json   ← view_axis → run_folder --view-gate
│  │  ├─ bust/               <stem>.json   ← bust_measure → rederive
│  │  ├─ depth/              <stem>.png    ← depth_prepass
│  │  └─ profiles/           <stem>.json   ← image_profile
│  ├─ _eval/                 FINDINGS.md + hand labels; not imported
│  ├─ comfy_nodes/amodal_prep/  the Amodal* node pack, versioned with the repo
│  ├─ comfy_nodes.yaml       registers that folder as a custom_nodes path
│  └─ README.md
├─ ComfyUI/                   runtime and weights; no pipeline source lives here
├─ start_server.bat          launch ComfyUI on :8199 (logs to server.log)
└─ run_nvidia_gpu.bat        stock portable launcher (port 8188, no ck-attention)
```

## Models

| role | file |
|---|---|
| diffusion (GGUF) | `Qwen-Rapid-NSFW-v23_Q3_K.gguf` |
| text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | `qwen_image_vae.safetensors` |
| inpaint ControlNet | `qwen_image_inpaint_diffsynth_controlnet.safetensors` |
| segmentation | `sam3.1_multiplex_fp16.safetensors` |
| matting | `birefnet.safetensors` |
| depth | `depth_anything_3_base.safetensors` (in `models/geometry_estimation/`) |
| flat-chest LoRA | `Eva_Flat_Chest.safetensors` (in `models/loras/`) — auto-applied below 0.50 bust, back views included |
| chest detector | `models/breast_size_y11m.pt` (in this repo) |
| booru tagger | `models/wd-swinv2-tagger-v3.onnx` (in this repo, for `view_axis.py`) |
| view gate VLM | `Qwen/Qwen3-VL-4B-Instruct` (HF cache, ~8.6 GB, only loaded when the tagger flags an image) |

They are read through `ComfyUI/extra_model_paths.yaml`, so the weights can live
in another ComfyUI install.

---

`adr/` (local, gitignored) holds the numbered decision records: what was decided,
the measurements behind it, and what was rejected. `adr/README.md` is the index.

## Quick start

```powershell
# 1. in its own window, leave it running
cd C:\Users\user\Desktop\SDXL\ComfyUI_windows_portable_v043
.\start_server.bat            # or the detached Start-Process in Troubleshooting

# 2. only when you added new images
cd C:\Users\user\Desktop\SDXL\ComfyUI_windows_portable_v043\sprite-amodal-completion-pipeline
..\python_embeded\python.exe depth_prepass.py
..\python_embeded\python.exe vlm_axes.py
..\python_embeded\python.exe bust_measure.py
..\python_embeded\python.exe view_axis.py       # writes output/axes/_views.json
..\python_embeded\python.exe rederive.py --refit

# 3. render
..\python_embeded\python.exe run_folder.py --view-gate --axes --no-profile --chunk 7 --force
```

Back-facing sources take the back branch when the gate fires. The one back the
gate misses — 0313, where the 4B reads a chest — goes on the list by hand:

```powershell
..\python_embeded\python.exe run_folder.py --assume-back 0313 --view-gate --axes --no-profile --chunk 7 --force
```

`start_server.bat` blocks its window, so step 2 and 3 need a second one. The
script checks for a live server before submitting and prints the start command
if there isn't one.

Outputs land in `output/<stem>_amodal.png`, same pixel size as the
source, RGBA with the original alpha preserved.

## Pipeline

```
source PNG
   │
   ├─ AmodalSafeFrame      canvas = source × 1.25, capped at 1.5 MP, aspect clamped
   ├─ AmodalDetectMany     one SAM3 backbone pass, then five prompts
   │                       (outfit · garments · face · hands · footwear)
   ├─ AmodalMaskUnionFill  fill holes per component, grow 16 px
   ├─ AmodalTextEncode     prompt + reference latent at the exact sampling size
   ├─ KSampler             v23, 4 steps, euler_ancestral/beta, cfg 1.0
   ├─ AmodalFinalize       composite back, resize to source size, restore alpha
   └─ BiRefNet             clean alpha
```

The pre-pass scripts are separate because they are expensive and cached:

| script | writes | consumed by |
|---|---|---|
| `depth_prepass.py` | `output/depth/<stem>.png` | `image_profile.py` |
| `vlm_axes.py` | `output/axes/<stem>.json` | `rederive.py` → `run_folder.py` |
| `bust_measure.py` | `output/bust/<stem>.json` | `rederive.py` |
| `view_axis.py` | `output/axes/_views.json` | `run_folder.py --view-gate` |
| `rederive.py` | `parts_cal` + `style` in the same JSON | `run_folder.py --axes` |
| `image_profile.py` | `output/profiles/<stem>.json` | `run_folder.py` (off with `--no-profile`) |

Everything is keyed by **image stem**, never by character: the same character
redrawn older, taller or in a different outfit gets its own parameters.

---

## Script reference

### `run_folder.py`

```powershell
..\python_embeded\python.exe run_folder.py --axes --no-profile --chunk 7 --force
```

| flag | default | meaning |
|---|---|---|
| `--dir` | `.` | subfolder of `input/`; default is `input/` itself |
| `--chunk` | 6 | images per ComfyUI prompt (model loads once per chunk) |
| `--steps` | 4 | v23 is a 4-step model; 6 for A/B only |
| `--seed` | 123456789 | sampler seed |
| `--limit` | 0 | stop after N images |
| `--force` | off | re-render even if `_amodal.png` exists |
| `--only` | — | substring filter on filenames |
| `--axes` | off | append `parts_cal` + `style` from `output/axes` |
| `--view-gate` | off | back views (tagger) get the back branch; the chest clause still describes the measured size, the low-bust tags stay off and the LoRA still applies |
| `--assume-back` | — | stems to force onto the back branch (substring match) |
| `--back-denoise` | 1.0 | denoise for back views; no-op above 0.80 at 4 steps (ADR-0008), A/B only |
| `--no-back-low` | off | back views: skip the low-bust LoRA and tags entirely (A/B) |
| `--back-low-tags` | off | back views: also append the pettanko tags below 0.26 (asserts a front chest) |
| `--no-profile` | off | skip the `image_profile` sentence |
| `--max-mp` | 1.5 | canvas megapixel cap (`1.0` ≈ 33 % faster) |
| `--sam-refine` | 2 | SAM3 refinement iterations per probe |
| `--sampler` / `--scheduler` | `euler_ancestral` / `beta` | |
| `--tag` | — | output suffix, e.g. `_v19` |
| `--parse` | off | add the segformer clothing mask |
| `--hair-protect` | off | subtract the SAM3 hair mask |
| `--no-footwear-protect` | off | stop protecting shoes/boots |
| `--no-equipment-protect` | off | stop protecting weapons, bags and worn gear; the equipment mask is dropped above 15% of the canvas (ADR-0012) |
| `--denoise` | 1.0 | sampler denoise inside the clothing mask |
| `--grow` | 16 | mask dilation, px |
| `--protect-threshold` | — | SAM3 score threshold for protected regions |
| `--prompt-extra` | — | text appended to the body prompt |
| `--garments` | — | override the SAM clothing prompt |
| `--unet` | — | override the GGUF unet name |
| `--legacy` | off | stock `TextEncodeQwenImageEditPlus` (A/B) |
| `--dump-mask` | off | save the final mask |
| `--lora` | — | patch a LoRA onto the unet, for every image |
| `--lora-strength` | 1.0 | strength for `--lora` |
| `--no-low-lora` | off | skip the automatic flat-chest LoRA below 0.50 bust |
| `--no-probe-merge` | off | five separate SAM3 detector nodes, for A/B (ADR-0012) |
| `--no-stage-order` | off | schedule image-by-image instead of stage-by-stage (ADR-0013) |
| `--no-inplace-patch` | off | clone the model for the ControlNet patch, stock node (ADR-0014) |
| `--group-lora` | off | sort LoRA variants together; measured neutral here (ADR-0014) |
| `--sync` | off | regenerate the UI workflow at startup |

### `vlm_axes.py`

12 forced-choice axes × 3 shuffled option orders, scored as a softmax over the
answer letters. Shuffling cancels the model's position bias; averaging is done
back in canonical order. Results are cached per image, so re-runs are free (a
cache written by an older axis set is detected and rescored).

`build` is not one question. Asked as "How is her figure drawn overall?" it
ranked 0.53 against the hand ranking; asked as two concrete questions — mass
against a typical heroine, and upper-arm thickness against head width — and
averaged, it ranks **~0.70** (0.737 and 0.675 on two independent shuffle
draws).

Three axes were measured and then removed rather than tuned. `limb` was never
read by any formula. `age_imp` scored 16/35 images under 0.25 ("childlike") and
pushed that noise into 27/35 prompts; the adult assertion belongs in
`comfy_client.POS_TXT`, where it already is. `hip_bias` put 30 of 35 images in
one bin — 150 testable pairs, no usable resolution — and the "shoulder frame"
term went with it.

The numbers behind all three decisions are in `_eval/FINDINGS.md`.

```powershell
..\python_embeded\python.exe vlm_axes.py            # ~3 s/image, ~8 GB peak
..\python_embeded\python.exe vlm_axes.py --rows 7   # less VRAM
..\python_embeded\python.exe vlm_axes.py --side 384 # fewer vision tokens
```

### `bust_measure.py`

Chest size does **not** come from the VLM. Asked directly, Qwen2.5-VL-3B answers
"small" or "medium" for almost every image; its letter-softmax probabilities are
prior-dominated and its generated answers are near-constant. The best VLM wording
reached 0.69 pairwise concordance on the reference set, and it got *worse* as the
size difference grew — it was ranking how much volume the outfit occupies, not
the chest. Loose clothing read as "large".

`bust_measure.py` instead runs Anzhc's *Breast size det cls v8* (YOLO11m, 15
classes, ~16k annotated illustrations, sizes relative to body proportion rather
than to the scene). Two views are averaged: the whole figure at 960 px and the
torso crop at 640 px, which is what recovers images whose chest is partly hidden
by an arm, a prop or loose clothing.

| | pairwise concordance vs the hand ranking |
|---|---|
| best VLM question | 0.69 |
| detector, single full view | 0.74 |
| detector, full + torso averaged | **0.91** (0.905 held out, 5-fold) |

```powershell
..\python_embeded\python.exe bust_measure.py           # writes output/bust/<stem>.json
..\python_embeded\python.exe bust_measure.py --report  # print the ranking only
```

Output is `bust` on 0..1, where 0.5 is the anime "average" of C/D cup. `rederive.py`
overrides the VLM `bust` axis with it wherever a measurement exists.

### `view_axis.py`

Decides which sources are back-facing, so `run_folder.py --view-gate` can stop
asserting a chest that is not visible. One instrument: WD SwinV2 v3, floor
`from_behind >= 0.01`.

| image | `from_behind` | truth |
|---|---|---|
| 0228 | 0.597 | back |
| 0313 | 0.488 | back |
| 0941 | 0.462 | back |
| 0584 | 0.436 | back |
| 0450 | 0.374 | back |
| 0878 | 0.263 | back (3/4) |
| 0479 | 0.083 | back |
| 0958 | 0.019 | back (3/4) |
| 0390 | 0.011 | mixed (upper back, hips front) |
| all other fronts | <= 0.003 | front |

8 of 8 backs, no fronts, 1 of 1 mixed. The whole safety margin is 0.019 -> 0.003,
so the floor is fitted to this corpus and has to be re-read when a new batch
lands. `_views.json` keeps the score per stem, so that is a one-line check.

The VLM is **not** consulted. Qwen3-VL-4B finds only 5 of the 8 backs (0313,
0450, 0941 read as chests) and its one front-side "back" (0653) is already below
the tagger floor; the 8B scores the same 5 of 8, and Thinking buys nothing. So
the conjunction was strictly worse than the tagger alone. `--vlm` still records
its answer for inspection.

```json
"0479_5617_xvgcgmwruygk_v83": {"view": "back", "candidate": true,
                               "from_behind": 0.083, "breasts": 0.021, "vlm": null}
```

```powershell
..\python_embeded\python.exe view_axis.py            # all sources
..\python_embeded\python.exe view_axis.py --only 0479 --force
..\python_embeded\python.exe view_axis.py --vlm      # also record the 4B (~8.6 GB)
```

A new "back" hit is a hint worth one look, not ground truth: four positives is
not a calibrated classifier.

### `rederive.py`

The only place a body prompt is built. Chest size is always emitted; frame comes
from `build`/`proportion`; at most 3 anatomy terms, one per family; opposing
pairs are resolved; duplicates removed.

`build` is ranked across the corpus before use (percentile, stretched to
0.15–0.85), so 0.5 is the median image and the fixed cuts in `frame_terms()`
mean what they say: 0.24 is the bottom ~13% ("very slender frame"), 0.34 the
bottom ~27% ("slender frame"). The same calibrated value feeds every formula;
previously `frame_terms()` read the raw axis while the derived terms read a
stretched one, and the cuts fired on the wrong images.

Chest phrasing is a bank: six size bands, four phrasings each. The phrasing is
picked from a hash of the image stem, so the same file always renders the same
words while different images do not. `--chest-variant N` forces one phrasing
across a run, which is how you A/B them.

```powershell
..\python_embeded\python.exe rederive.py                   # use frozen calibration
..\python_embeded\python.exe rederive.py --refit           # refit on the current corpus
..\python_embeded\python.exe rederive.py --chest-variant 1 # force phrasing 1
```

`output/axes/_calibration.json` is a global percentile ruler (10th–90th), not a
per-character anchor. It is fitted on whatever images are present, so adding
images shifts every cut — rerun with `--refit` whenever the corpus changes.

### `depth_prepass.py`

DA3 depth maps for `image_profile.py`. Only needed if you use `image_profile`.

```powershell
..\python_embeded\python.exe depth_prepass.py --resolution 504
```

### `image_profile.py`

A measured per-image body sentence from depth + segmentation. Off by default in
the batch (`--no-profile`), because the axis fragment is the tuned path.

```powershell
..\python_embeded\python.exe image_profile.py [--force]
```

### `evaluate.py`

Compares every source against `<stem>_amodal.png`, prints silhouette IoU, alpha
delta and mean pixel diff, then writes `output/sheet_<n>.png` contact sheets.
`--no-sheets` prints the table only.

---

## Paths

`config.py` is the single source of truth:

```python
_env_root = os.environ.get("FLOW_ROOT")
ROOT   = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parent.parent
HERE   = Path(__file__).resolve().parent
COMFY  = ROOT / "ComfyUI"
INPUT  = str(HERE / "input")                  # sources, flat
OUTPUT = str(HERE / "output")                 # <stem>_amodal.png
AXES   = str(HERE / "output" / "axes")        # BUST / DEPTH / PROFILES beside it
HOST   = "http://127.0.0.1:8199"
```

`FLOW_ROOT` overrides `ROOT` if the scripts live somewhere else.

---

## Performance (RTX 4070 Ti, 12 GB)

| stage | cost |
|---|---|
| `vlm_axes.py` | ~3 s/image, flat 8.0 GB peak |
| `run_folder.py`, 4 steps, chunk 7 | ~50 s/image |
| sampling (20B + ControlNet) | ~30–35 s of that |

The 12 GB card cannot keep unet, text encoder and SAM3 resident together, so
loads happen constantly. `--max-mp 1.0` and `--sam-refine 1` are the remaining
safe levers.

That is a hard constraint, not an observation: a lever that only works with more
VRAM is not a solution. The pose ControlNet is the worked example — 3.5 GB of
weights beside the 9.6 GB unet, and swap, CPU offload and fp8 each ended slower
than the baseline or broken (`_eval/FINDINGS.md` section 15).

Chunking only helps *because* the default stage ordering runs detect-all,
encode-all, sample-all (ADR-0013): before that, every image in a chunk reloaded
the text encoder and unet, and bigger chunks amortised nothing. Measured on
three images at chunk 3, 4 steps, cold: **91 s/image before, 61 s/image after**.
The unet still reloads once per image because its ControlNet patch is per-image.

Detection is cached across seeds and step counts, so it is only paid on the first
render of an image.

---

## Troubleshooting

**`ConnectionRefusedError: [WinError 10061]`** — ComfyUI is not running.
`run_folder.py` checks first and prints the start command. Start it with
`..\start_server.bat`, or detached:

```powershell
Start-Process -WindowStyle Hidden -FilePath "..\python_embeded\python.exe" `
  -ArgumentList "-s","..\ComfyUI\main.py","--windows-standalone-build","--enable-manager",
                "--port","8199","--listen","127.0.0.1","--use-ck-attention",
                "--input-directory","$PWD\input","--output-directory","$PWD\output",
                "--extra-model-paths-config","$PWD\comfy_nodes.yaml" `
  -RedirectStandardOutput "..\server.log" -RedirectStandardError "..\server.err.log"
```

The three flags are required: `config.py` points at this repo's `input/` and
`output/`, and without them ComfyUI reads its own folders and every render fails
to find its source. Without the yaml, every `Amodal*` node is "not found".

**Batch finishes in seconds and nothing changed** — ComfyUI cached the result.
Changing only `filename_prefix` reuses the sampler; use `--force` or a different
`--tag`.

**GPU slows to a crawl, shared GPU memory grows** — Windows is paging VRAM.
Peak memory must stay under ~11 GB. For `vlm_axes.py` use `--rows 7`; for the
batch use a smaller `--max-mp` or `--chunk`.

**`OSError: [Errno 22]` from a progress bar** — ComfyUI was started with stderr
attached to a console that went away. `start_server.bat` redirects stderr to
`server.log`; keep that.

**`ImportError: No module named 'config'`** — the portable python uses a `._pth`
file, so the working directory is not on `sys.path`. Run the scripts by path
(`..\python_embeded\python.exe run_folder.py ...`), not with `-c`.

---

## Notes

- **The chest bias on flat sources is a model floor, not a prompt bug.** Measured
  with the detector on 35 renders: flat sources come out at class 1.0-2.0
  ("small" to "medium") against a source class of 0.0, i.e. +1.3 classes; large
  and huge sources are accurate (+0.05, 0.00). Rewriting every chest phrasing
  moved the output class by **0.00**; denoise 1.0 -> 0.90 also 0.00, 0.80 gives
  -0.15; the inpaint ControlNet is a region marker, not a shape constraint.
  What does help, slightly: never asserting "bare breasts" in the base prompt,
  and appending the booru tags `pettanko, tiny breasts, flat chest` for
  `bust < 0.26` (0201 1.61 -> 1.22, 0381 1.15 -> 1.00, mean -0.14). `0231` and
  `0248` do not move under any setting tried. Run-to-run variance is +-0.13
  class, so treat anything smaller than that as noise.
- **Text is load-bearing but saturates at one phrase.** Same image, three prompts:
  no chest words at all renders at class 1.90, any flat wording at 1.09-1.14, and
  a single bare tag `flat chest` is as good as a three-phrase clause. So the
  words buy 0.75 of the 1.9 prior and nothing beyond that — do not try to prompt
  the rest away.
- **A flat-chest LoRA does the rest, partially.** `Eva_Flat_Chest` (trained on
  Qwen-Image, not Edit, so cross-base) at strength 1.0, applied below 0.26 bust
  at the time of this measurement (the gate has since moved to 0.50; ADR-0008):
  mean bias **+1.31 -> +0.52** over 20 images, 15 improved, 5
  unchanged, none worse. Strength 0.7 does nothing — the dose response is a
  cliff. Cost is ~0 (74.3 vs 73.5 s/image at chunk 6-7). The trigger word must
  come *before* the other chest phrases: after them, 0381 renders 0.91 instead
  of 0.30.
- The `Amodal*` node class names and the `amodal_prep` folder name are registered
  identifiers. Saved workflows reference them as strings, so renaming breaks
  every existing graph; regenerate with `--sync` after a rename. The names say
  what the nodes do (complete the masked region), not what the render contains —
  that is what keeps them distinguishable from the detector packs (`NudeNet` and
  friends) that dominate the `Nude*` namespace.
- The node pack lives in `comfy_nodes/amodal_prep` inside this repo, not in
  `ComfyUI/custom_nodes`. `comfy_nodes.yaml` registers that folder as a second
  `custom_nodes` path and `start_server.bat` passes it with
  `--extra-model-paths-config`. The yaml must point at a folder that holds only
  node packs — ComfyUI also imports every top-level `*.py` in a registered path,
  so pointing it at the repo root would import `config.py` and `run_folder.py`
  at startup.
- `KSampler` runs at `cfg 1.0`, so the negative conditioning is ignored — every
  constraint has to be a positive assertion.
- Text is a weak lever: the torso is masked at denoise 1.0, so the model rebuilds
  it from its prior. Prompt changes shift chest size a little; they do not flatten
  it. A low-denoise torso pass is the remaining geometric lever.
- `PARSE = False` and `HAIR_PROTECT = False` in `run_folder.py` are deliberate;
  both were measured to hurt stylised costumes. See the comments there.
