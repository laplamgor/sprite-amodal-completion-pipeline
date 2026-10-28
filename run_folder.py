"""Process every image in a folder using the same nodes as the UI workflow.

One ComfyUI prompt per chunk, so the 20B model and the text encoder load once per
chunk. Output keeps the source name: <stem>_amodal.png in ComfyUI/output.

    python run_folder.py --dir incoming --chunk 6
"""
import os
import sys
import glob
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C          # noqa: E402
import comfy_client as t  # noqa: E402

INPUT = C.INPUT
OUTPUT = C.OUTPUT

UNET = "Qwen-Rapid-NSFW-v19_Q3_K.gguf"
TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
CN = "qwen_image_inpaint_diffsynth_controlnet.safetensors"
SAM = "sam3.1_multiplex_fp16.safetensors"
BIREF = "birefnet.safetensors"

SAMPLER = "euler_ancestral"
SCHEDULER = "beta"
STEPS = 4          # v19 is trained as a 4-step model; 6 only for A/B comparison

# Mask policy (see make_ui_workflow, which reads these too).
# PARSE: segformer clothing mask as an extra source. Off: it over-segments stylised
#        costumes (1003_5554 came out as one "dress" over the whole body).
# HAIR_PROTECT: subtract the SAM3 hair mask from the clothing mask. Off: SAM3's hair
#        mask is loose enough to swallow sleeves that sit under long hair, which left
#        half a polo shirt on 1005_4485. Hair survives without it because the clothing
#        mask does not actually cover hair.
PARSE = False
HAIR_PROTECT = False
FOOTWEAR_PROTECT = True
PROTECT_THRESHOLD = 0.25

PROMPT = ("Remove all of her clothing. She is completely nude, bare torso, "
          "bare arms, bare legs, bare feet, bare hands. Keep the exact same character, same face, "
          "same hairstyle, same hair colour, same pose, same arm positions, same body shape, "
          "same camera angle. Anime illustration, same art style, flat plain solid colour background.")

# The model answers booru tags far better than English descriptions, and the
# chest fragment is written in English.  For the two low bands, append the tags
# under the base prompt (before the body fragment, where the tag register still
# applies).  Measured: 0201 1.61 -> 1.29 class; no image got worse.
LOW_BUST_BELOW = 0.26
LOW_BUST_TAGS = " pettanko, tiny breasts, flat chest."

# LoRA for the two lowest bands.  Eva_Flat_Chest is trained on Qwen-Image, not
# Edit, but it still works on this merge: at strength 1.0 it pulled 0381 from
# class 1.00 to 0.30 and 0201 from 1.22 to 0.77, with face/hair/props intact.
# At 0.7 it did nothing -- the dose response is a cliff, not a slope.  Gated
# because it is a global LoRA and would also shrink the large ones.
# Cost: +3.3 s/image (~6.5%) for the one-time patch of the 9.4 GB unet.
LOW_LORA = "Eva_Flat_Chest.safetensors"
LOW_LORA_STRENGTH = 1.0
LOW_LORA_TRIGGER = " Eva_Flat_Chest"
LOW_LORA_BELOW = 0.26


def discover(folder):
    base = folder if os.path.isabs(folder) else os.path.join(INPUT, folder)
    files = sorted(glob.glob(os.path.join(base, "*.png")))
    out = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.endswith("_amodal"):
            continue  # never treat our own outputs as inputs
        rel = os.path.relpath(f, INPUT).replace("\\", "/")
        out.append((rel, stem[:40]))
    return out


# NOTE: do NOT add "arm"/"glove"/"hand" here. They pull limbs into the mask and the
# model sometimes fails to redraw them (0994 lost an entire arm). Use --garments for
# outliers such as heavily cloaked characters.
# "shirt / top / collar" were missing for a long time: the white polo on 1005 leaked
# through because only the generic "outfit" probe caught it.
GARMENTS = ("shirt, top, collar, polo, dress, skirt, bra, hood, cloak, cape, robe, "
            "ribbon, sash, belt, apron, swimsuit, bikini")


def branch(nid, rel, stem, seed, steps, garments=None, sampler="euler", scheduler="beta",
           tag="", legacy=False, dump_mask=False, grow=16, parse=True,
           protect_threshold=0.25, hair_protect=True, footwear_protect=True,
           denoise=1.0, prompt_extra="", max_mp=1.5, sam_refine=2,
           lora=None, lora_strength=1.0):
    """One image branch; returns (nodes, save_node_id).

    The clothing mask is protected: face / hair / hand masks are detected and
    SUBTRACTED, so the model can never repaint the mouth, a hand or the hair even
    when a garment term overlaps them.

    `lora` patches the unet before the ControlNet, so the LoRA also applies to
    the inpaint pass; 40 is free because nid advances by 100 per image.
    """
    model_ref = [str(nid + 40), 0] if lora else ["901", 0]
    n = {
        str(nid): {"class_type": "LoadImage", "inputs": {"image": rel}},
        str(nid + 1): {"class_type": "NudeSafeFrame", "inputs": {
            "image": [str(nid), 0], "mask": [str(nid), 1],
            "margin": 0.08, "scale": 1.25, "max_megapixels": max_mp,
            "min_aspect": 0.62, "max_aspect": 1.60}},
        str(nid + 2): {"class_type": "CLIPTextEncode", "inputs": {"text": "outfit", "clip": ["900", 1]}},
        str(nid + 3): {"class_type": "CLIPTextEncode", "inputs": {"text": garments or GARMENTS, "clip": ["900", 1]}},
        str(nid + 4): {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0], "conditioning": [str(nid + 2), 0],
            "threshold": 0.30, "refine_iterations": sam_refine, "individual_masks": False}},
        str(nid + 5): {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0], "conditioning": [str(nid + 3), 0],
            "threshold": 0.30, "refine_iterations": sam_refine, "individual_masks": False}},
        str(nid + 6): {"class_type": "NudeMaskUnionFill", "inputs": {
            "mask1": [str(nid + 4), 0], "mask2": [str(nid + 5), 0], "grow": grow, "min_area": 400}},

        # --- protected features ---
        str(nid + 7): {"class_type": "CLIPTextEncode", "inputs": {"text": "face, mouth, lips", "clip": ["900", 1]}},
        str(nid + 8): {"class_type": "CLIPTextEncode", "inputs": {"text": "hair, braid", "clip": ["900", 1]}},
        str(nid + 9): {"class_type": "CLIPTextEncode", "inputs": {"text": "hand, fingers", "clip": ["900", 1]}},
        str(nid + 10): {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0], "conditioning": [str(nid + 7), 0],
            "threshold": protect_threshold, "refine_iterations": sam_refine, "individual_masks": False}},
        str(nid + 11): {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0], "conditioning": [str(nid + 8), 0],
            "threshold": protect_threshold, "refine_iterations": sam_refine, "individual_masks": False}},
        str(nid + 12): {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0], "conditioning": [str(nid + 9), 0],
            "threshold": protect_threshold, "refine_iterations": sam_refine, "individual_masks": False}},
        str(nid + 13): {"class_type": "NudeMaskUnionFill", "inputs": {
            "mask1": [str(nid + 10), 0], "mask2": [str(nid + 11), 0], "mask3": [str(nid + 12), 0],
            "grow": 4, "min_area": 100}},
        str(nid + 27): {"class_type": "NudeParseClothes", "inputs": {
            "image": [str(nid + 1), 0], "grow": 6}},
        str(nid + 28): {"class_type": "MaskComposite", "inputs": {
            "destination": [str(nid + 6), 0], "source": [str(nid + 27), 0],
            "x": 0, "y": 0, "operation": "add"}},
        str(nid + 14): {"class_type": "MaskComposite", "inputs": {
            "destination": [str(nid + 28), 0], "source": [str(nid + 13), 0],
            "x": 0, "y": 0, "operation": "subtract"}},
        str(nid + 15): {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
            "model": model_ref, "model_patch": ["903", 0], "vae": ["902", 0],
            "image": [str(nid + 1), 0], "mask": [str(nid + 14), 0],
            "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
        str(nid + 16): ({"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["904", 0], "prompt": PROMPT + prompt_extra, "vae": ["902", 0],
            "image1": [str(nid + 1), 0]}} if legacy else
            {"class_type": "NudeTextEncode", "inputs": {
            "clip": ["904", 0], "prompt": PROMPT + prompt_extra, "vae": ["902", 0],
            "image1": [str(nid + 1), 0], "latent": [str(nid + 17), 0]}}),
        str(nid + 17): {"class_type": "VAEEncode", "inputs": {"pixels": [str(nid + 1), 0], "vae": ["902", 0]}},
        str(nid + 18): {"class_type": "SetLatentNoiseMask", "inputs": {
            "samples": [str(nid + 17), 0], "mask": [str(nid + 14), 0]}},
        str(nid + 19): {"class_type": "KSampler", "inputs": {
            "model": [str(nid + 15), 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": sampler, "scheduler": scheduler,
            "positive": [str(nid + 16), 0], "negative": [str(nid + 16), 0],
            "latent_image": [str(nid + 18), 0], "denoise": denoise}},
        str(nid + 20): {"class_type": "VAEDecode", "inputs": {"samples": [str(nid + 19), 0], "vae": ["902", 0]}},
        str(nid + 21): {"class_type": "RemoveBackground", "inputs": {
            "bg_removal_model": ["905", 0], "image": [str(nid + 20), 0]}},
        str(nid + 22): {"class_type": "NudeFinalize", "inputs": {
            "image": [str(nid + 20), 0], "mask": [str(nid + 21), 0], "frame": [str(nid + 1), 1],
            "source_image": [str(nid), 0], "source_mask": [str(nid), 1],
            "edited_mask": [str(nid + 14), 0],
            "edge_soften": 0.6, "soft_alpha_below": 0.98, "bg_tolerance": 18.0}},
        str(nid + 23): {"class_type": "JoinImageWithAlpha", "inputs": {
            "image": [str(nid + 22), 0], "alpha": [str(nid + 22), 1]}},
        str(nid + 24): {"class_type": "SaveImage", "inputs": {
            "images": [str(nid + 23), 0], "filename_prefix": stem + "_amodal" + tag}},
    }
    if lora:
        n[str(nid + 40)] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["901", 0], "lora_name": lora,
            "strength_model": lora_strength}}
    # protection union: face always; hair / hands / footwear optional
    prot_ids = [nid + 10]                      # face, mouth, lips
    if hair_protect:
        prot_ids.append(nid + 11)
    else:
        n.pop(str(nid + 11), None)
    prot_ids.append(nid + 12)                  # hand, fingers
    if footwear_protect:
        n[str(nid + 33)] = {"class_type": "CLIPTextEncode",
                            "inputs": {"text": "shoe, boot, footwear, sandal",
                                       "clip": ["900", 1]}}
        n[str(nid + 34)] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0],
            "conditioning": [str(nid + 33), 0],
            "threshold": protect_threshold, "refine_iterations": sam_refine,
            "individual_masks": False}}
        prot_ids.append(nid + 34)
    ins = {"mask%d" % (i + 1): [str(x), 0] for i, x in enumerate(prot_ids)}
    ins.update({"grow": 4, "min_area": 100})
    n[str(nid + 13)]["inputs"] = ins
    if not parse:
        n[str(nid + 14)]["inputs"]["destination"] = [str(nid + 6), 0]
        n.pop(str(nid + 27), None)
        n.pop(str(nid + 28), None)
    if dump_mask:
        n[str(nid + 25)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(nid + 14), 0]}}
        n[str(nid + 26)] = {"class_type": "SaveImage", "inputs": {
            "images": [str(nid + 25), 0], "filename_prefix": stem + "_mask" + tag}}
        n[str(nid + 29)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(nid + 13), 0]}}
        n[str(nid + 30)] = {"class_type": "SaveImage", "inputs": {
            "images": [str(nid + 29), 0], "filename_prefix": stem + "_protect" + tag}}
        if parse:
            n[str(nid + 31)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(nid + 27), 0]}}
            n[str(nid + 32)] = {"class_type": "SaveImage", "inputs": {
                "images": [str(nid + 31), 0], "filename_prefix": stem + "_parse" + tag}}
    return n, str(nid + 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="incoming")
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="substring filters on filename")
    ap.add_argument("--garments", default=None, help="override the SAM clothing prompt")
    ap.add_argument("--unet", default=None, help="override the GGUF unet name")
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--scheduler", default=None)
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _v19")
    ap.add_argument("--legacy", action="store_true",
                    help="use the stock TextEncodeQwenImageEditPlus (for A/B baselines)")
    ap.add_argument("--dump-mask", action="store_true",
                    help="also save the final clothing mask for debugging")
    ap.add_argument("--grow", type=int, default=16,
                    help="clothing mask dilation in px (default 16)")
    ap.add_argument("--parse", action="store_true", default=PARSE,
                    help="add the segformer clothing mask (off by default: it over-segments "
                         "stylised costumes, e.g. 1003_5554)")
    ap.add_argument("--protect-threshold", type=float, default=PROTECT_THRESHOLD,
                    help="SAM3 threshold for face/hair/hand protection (default 0.25)")
    ap.add_argument("--hair-protect", action="store_true", default=HAIR_PROTECT,
                    help="subtract the SAM3 hair mask from the clothing mask (off by default: "
                         "it swallows sleeves that sit under long hair)")
    ap.add_argument("--no-footwear-protect", action="store_true",
                    help="stop protecting shoes/boots from the clothing mask")
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="sampler denoise inside the clothing mask (1.0 = repaint fully)")
    ap.add_argument("--prompt-extra", default="",
                    help="extra sentence appended to the body prompt (trial only)")
    ap.add_argument("--no-profile", action="store_true",
                    help="do not append the per-character body profile to the prompt")
    ap.add_argument("--axes", action="store_true",
                    help="append the VLM axis fragment from input/axes/<stem>.json")
    ap.add_argument("--max-mp", type=float, default=1.5,
                    help="canvas megapixel cap; 1.0 is ~33%% faster and barely differs")
    ap.add_argument("--sam-refine", type=int, default=2,
                    help="SAM3 refinement iterations per probe (1 is ~half the cost)")
    ap.add_argument("--lora", default=None,
                    help="LoRA filename in models/loras, patched onto the unet")
    ap.add_argument("--lora-strength", type=float, default=1.0,
                    help="strength for --lora")
    ap.add_argument("--no-low-lora", action="store_true",
                    help="do not auto-apply the flat-chest LoRA below 0.26 bust")
    ap.add_argument("--no-sync", action="store_true", help="skip regenerating the UI workflow")
    a = ap.parse_args()
    sampler = a.sampler or SAMPLER
    scheduler = a.scheduler or SCHEDULER

    # fail fast with a usable message instead of a urllib traceback mid-run
    try:
        import urllib.request as _u
        with _u.urlopen(t.HOST + "/system_stats", timeout=5):
            pass
    except Exception:
        raise SystemExit(
            "ComfyUI is not responding at %s.\n\n"
            "Start it, then wait ~15 s before re-running:\n"
            "  ..\\start_server.bat                 (foreground, logs to server.log)\n\n"
            "or detached, from this folder:\n"
            "  Start-Process -WindowStyle Hidden -FilePath \"..\\python_embeded\\python.exe\" `\n"
            "    -ArgumentList \"-s\",\"..\\ComfyUI\\main.py\",\"--windows-standalone-build\","
            "\"--enable-manager\",\n"
            "                  \"--port\",\"8199\",\"--listen\",\"127.0.0.1\","
            "\"--use-ck-attention\" `\n"
            "    -RedirectStandardOutput \"..\\server.log\" "
            "-RedirectStandardError \"..\\server.err.log\"\n\n"
            "Check with:\n"
            "  Invoke-WebRequest http://127.0.0.1:8199/system_stats -UseBasicParsing | "
            "Select-Object StatusCode" % t.HOST)

    if not a.no_sync:
        try:
            import make_ui_workflow as muw
            out, n, l = muw.write()
            print(f"synced UI workflow: {os.path.basename(out)} ({n} nodes, {l} links)")
        except Exception as e:
            print("workflow sync skipped:", e)

    files = discover(a.dir)
    todo = []
    for rel, stem in files:
        out = os.path.join(OUTPUT, stem + "_amodal" + a.tag + ".png")
        src = os.path.join(INPUT, rel)
        if not a.force and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            continue
        todo.append((rel, stem))
    if a.limit:
        todo = todo[:a.limit]
    if a.only:
        todo = [t for t in todo if any(o.lower() in t[0].lower() for o in a.only)]

    print(f"found {len(files)}, todo {len(todo)}")
    if not todo:
        return

    t0 = time.time()
    low = []
    if a.axes and not a.no_low_lora and a.lora is None:
        import json as _json
        for rel, stem in todo:
            p = os.path.join(INPUT, "axes", stem + ".json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                b = _json.load(fh).get("bust_measured")
            if b is not None and b < LOW_LORA_BELOW:
                low.append(stem)
    if low:
        try:
            import urllib.request as _u
            info = _json.loads(_u.urlopen(
                t.HOST + "/object_info/LoraLoaderModelOnly", timeout=30).read())
            have = info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
        except Exception as e:
            print("cannot check LoRA list (%s), assuming present" % e)
            have = [LOW_LORA]
        if LOW_LORA not in have:
            print("!! %s not in models/loras -- %d low-bust images will render "
                  "WITHOUT it" % (LOW_LORA, len(low)))
            a.no_low_lora = True
        else:
            print("flat-chest LoRA %s -> %d low-bust image(s)"
                  % (LOW_LORA, len(low)))

    for ci in range(0, len(todo), a.chunk):
        chunk = todo[ci:ci + a.chunk]
        wf = {
            "900": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM}},
            "901": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": a.unet or UNET}},
            "902": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "903": {"class_type": "ModelPatchLoader", "inputs": {"name": CN}},
            "904": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "qwen_image", "device": "default"}},
            "905": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BIREF}},
        }
        saves = {}
        nid = 1000
        for rel, stem in chunk:
            extra = a.prompt_extra
            lora, lora_strength = a.lora, a.lora_strength
            if not a.no_profile:
                try:
                    import image_profile
                    extra = extra + image_profile.profile_for(
                        os.path.join(INPUT, rel))["prompt"]
                except Exception as e:
                    print("    profile skipped:", e)
            if a.axes:
                import json as _json
                apath = os.path.join(INPUT, "axes", stem + ".json")
                if os.path.exists(apath):
                    with open(apath, encoding="utf-8") as _fh:
                        rec = _json.load(_fh)
                    frag = rec.get("parts_cal") or rec.get("parts") or []
                    style = rec.get("style") or []
                    bust = rec.get("bust_measured")
                    if bust is not None and bust < LOW_BUST_BELOW:
                        if not a.no_low_lora and lora is None:
                            lora = LOW_LORA
                            lora_strength = LOW_LORA_STRENGTH
                            extra = extra + LOW_LORA_TRIGGER   # trigger first
                        extra = extra + LOW_BUST_TAGS
                    if frag or style:
                        extra = extra + " Her body: " + ", ".join(frag + style) + "."
                else:
                    print("    axes missing for", stem)
            nodes, save_id = branch(nid, rel, stem, a.seed, a.steps, a.garments,
                                    sampler, scheduler, a.tag, a.legacy, a.dump_mask,
                                    a.grow, a.parse, a.protect_threshold, a.hair_protect,
                                    not a.no_footwear_protect, a.denoise, extra,
                                    a.max_mp, a.sam_refine, lora, lora_strength)
            wf.update(nodes)
            saves[stem] = save_id
            nid += 100

        tc = time.time()
        r = t.post_prompt(wf)
        print(f"[chunk {ci//a.chunk+1}] {len(chunk)} images, submitted {r['prompt_id']}", r.get("node_errors"))
        h = t.wait_for(r["prompt_id"], timeout=14400)
        ok = 0
        import shutil
        for stem, sid in saves.items():
            if sid not in h.get("outputs", {}):
                continue
            for im in h["outputs"][sid]["images"]:
                src = os.path.join(OUTPUT, im["filename"])
                dst = os.path.join(OUTPUT, stem + "_amodal" + a.tag + ".png")
                if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                    for attempt in range(5):
                        try:
                            if os.path.exists(dst):
                                os.remove(dst)
                            shutil.move(src, dst)
                            break
                        except PermissionError:
                            import time as _t
                            _t.sleep(0.5)
                ok += 1
        print(f"    {ok}/{len(chunk)} ok in {time.time()-tc:.0f}s  (total {time.time()-t0:.0f}s)")

    print("DONE")


if __name__ == "__main__":
    main()
