# Project instructions — sprite-amodal-completion-pipeline

A batch pipeline that repaints garment regions of transparent-background anime
PNGs as bare skin, preserving pose, identity, hair, accessories and props.
Output must match source dimensions exactly and stay RGBA with soft alpha.

**Adults only.** `adult` stays in the prompt; never trade it away for a body hint.

## Read before changing anything

1. `README.md` — what every script and flag does.
2. `adr/README.md` — index of decision records. Each one is a decision *and* the
   measurements behind it, plus what was already rejected.

`adr/` is gitignored, so it exists only on this machine. If it is missing, say so
and ask — do not reconstruct its history from memory.

## Hard constraints

- **Fully automated.** No manual masking, no per-image prompt edits, no
  per-character anchoring. All parameters are keyed by *image stem*, never by
  character. The same file must always render the same way.
- Preserve hair, footwear, accessories, props, face, mouth and fingers. The mask
  is built by subtraction for exactly this reason.
- `KSampler` runs at `cfg 1.0`, so the negative conditioning is ignored — every
  constraint must be a positive assertion. A negative prompt cannot work here
  without a second text node *and* cfg > 1.
- Do not reintroduce hard-coded `C:\...` paths. `config.py` is the single source
  of truth; `FLOW_ROOT` overrides the root.

## Measurement discipline

Any claim about quality must be measured, and any decision that comes out of it
must be written down.

- Metric: **pairwise concordance** against the hand labels (`_eval/hand.json`,
  `_eval/truth.json`), cross-bin pairs only, **ties worth 0.5**, each unordered
  pair once. Always report the pair count — 391 pairs over 35 images is the
  working resolution, so differences under ~0.05 are noise.
- Run-to-run variance on one image is **+-0.13 class**. Anything smaller is not a
  result.
- The bust detector reports whole classes; an improvement below one class shows
  as 0.00. Check visually before concluding "no change".
- Prefer 5 pairwise comparisons to one eyeballed A/B, and prefer a same-image
  controlled test to a cross-image correlation.
- `0958_6353_zsnmjonjfpan_v79` is a misdrawn source (wrong pose). It stays in
  `incoming/` but is excluded from every evaluation; do not tune anything to fix
  its scores.

## Keeping the records current

- **Adding a decision?** Write `adr/NNNN-short-slug.md` from the template (Status,
  Date, Keywords, Touches, then Context / Decision / Evidence / Consequences) and
  add a row to the index in `adr/README.md`.
- **Changing an earlier decision?** Do not rewrite it silently. Mark the old one
  `Superseded by ADR-NNNN` and add the new record with the new evidence.
- **Rejecting something?** Record it with the numbers. Negative results are the
  most valuable records here — they stop the next session repeating them.
- `_eval/FINDINGS.md` is the raw experiment log; `adr/` holds the decisions. Write
  numbers in the ADR, not just a pointer.

## Already rejected — do not retry without a new reason

- Prompt wording for chest size: measured 0.00 effect across four phrasings and a
  full rewrite. Text saturates at one phrase (ADR-0008).
- Denoise below 1.0 to shrink the chest: 0.90 = 0.00, 0.80 = -0.15 (ADR-0008).
- Bigger VLM (4B) or thinking mode for the axes: worse on every axis, and the 4B
  answers identically for 27/35 images (ADR-0007).
- Geometry for `build`: 0.616 vs ~0.70, and stacking it lowered the score.
- `limb`, `age_imp`, `hip_bias`: removed for cause, with numbers (ADR-0004).

## Workflow

```powershell
# server: its own window, blocks, port 8199.  It lives one level up, next to this
# repo (you are in ..._v043\sprite-amodal-completion-pipeline).
..\start_server.bat

# only when new images were added
..\python_embeded\python.exe depth_prepass.py --dir incoming
..\python_embeded\python.exe vlm_axes.py
..\python_embeded\python.exe bust_measure.py
..\python_embeded\python.exe rederive.py --refit

# render, then verify
..\python_embeded\python.exe run_folder.py --dir incoming --axes --no-profile --chunk 7 --force
..\python_embeded\python.exe evaluate.py --no-sheets
```

- `rederive.py --refit` is required whenever the corpus changes: `build` is ranked
  across the corpus, so adding images moves every cut.
- The flat-chest LoRA is applied automatically below `bust < 0.26`; `--no-low-lora`
  disables it. The file is required and lives in the model host:
  `..\..\ComfyUI_windows_portable\ComfyUI\models\loras\Eva_Flat_Chest.safetensors`.
  `run_folder.py` checks the server's LoRA list at startup and warns rather than
  failing if it is absent.
- `--tag` is for A/B renders. Delete tagged outputs when the A/B is over; only
  `<stem>_amodal.png` is a real output.

## Style

Match the surrounding code: module docstrings explain *why* with the measured
reason, constants are named and commented with their evidence, and comments state
the trap rather than the obvious. No emoji, no exclamation marks.
