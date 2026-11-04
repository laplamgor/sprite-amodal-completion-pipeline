"""Process every image in a folder using the same nodes as the UI workflow.

One ComfyUI prompt per chunk, so the 20B model and the text encoder load once per
chunk. Output keeps the source name: <stem>_amodal.png in output/.

    python run_folder.py --chunk 6                    # all of input/
    python run_folder.py --dir incoming2 --chunk 6     # one side batch
"""
import os
import sys
import glob
import json
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
# EQUIPMENT_PROTECT subtracts props from the clothing mask. Without it the mask
# swallows whatever the character carries or wears on her back, and the model
# paints skin over it: 0313 -- a shipgirl with a rig on her back -- lost the rig,
# the gun in her left hand and the whole arm, leaving a white smear. Footwear
# protection is the same idea at the other end of the body, and this is the
# "preserve props" rule the pipeline already promises.
EQUIPMENT_PROTECT = True
# Word choice is measurement, not taste (per-word SAM3 mask area, % of canvas):
#
#   word       0002   0313   0958
#   armor      4.0    7.3    8.3   <-- 8.3 is 0958's haori: the coat stayed on
#   mecha     30.1    8.8    1.5   <-- 30.1 is 0002's whole figure: nothing removed
#   bag        5.0    7.2    1.7
#   backpack   1.0    8.3    1.5
#   machinery  4.8    1.5    2.0
#   equipment  0.3    1.5    2.0
#   gun        1.0    1.5    0.2
#
# A protection mask is subtracted, so a word that fires on the whole figure
# silently disables clothing removal (0002) and a word that fires on a garment
# keeps that garment (0958). armour and mecha are out for exactly those reasons.
EQUIPMENT = "weapon, gun, sword, shield, backpack, bag, machinery, equipment"
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

# Back-facing sources need a different branch: the front prompt asserts a chest
# the source does not show, and the low-bust tags assert one more loudly.  On
# 0479 -- torso away from the camera, head turned over the shoulder, long sleeves
# -- the front prompt rebuilt a front torso and swapped both arms at denoise 1.0.
#
# BACK_PROMPT keeps every "same ..." assertion and swaps only the anatomy the
# model is asked to invent: a bare back instead of a bare torso/breasts, plus an
# explicit "seen from behind" so the arms are rebuilt in the pose they are in.
#
# Routing is an explicit stem list (`--assume-back`), not a classifier.  Both
# instruments that were measured guess "back" for 0313, a front-facing shipgirl
# with a rack on her back and a sailor top over a small chest: WD tags give
# from_behind 0.49 / breasts 0.01, and Qwen2.5-VL-3B answers "We see her back."
# A false "back" repaints a front source backwards, which is worse than missing
# the one true back view in 22 images, so the label stays a human decision.
BACK_PROMPT = ("Remove all of her clothing. She is completely nude. Seen from behind: "
               "bare back, bare shoulders, bare arms, bare legs, bare feet, bare hands. "
               "Her torso is facing away from the camera, we see her back, not her chest. "
               "Keep the exact same character, same face, "
               "same hairstyle, same hair colour, same pose, same arm positions, same body shape, "
               "same camera angle. Anime illustration, same art style, flat plain solid colour background.")


def is_back(stem, views, assume=()):
    """True when the stem is on the explicit list or the consensus gate says back.

    `assume` entries match as substrings, like --only, so `--assume-back 0479`
    works without typing the whole stem.  The gate needs both the tagger and the
    VLM to agree (view_axis.py); a stem listed here always wins.
    """
    if any(a.lower() in stem.lower() for a in assume):
        return True
    return views.get(stem, {}).get("view") == "back"


VIEWS = os.path.join(C.AXES, "_views.json")

# On a back view the chest clause is normally kept and describes what the
# detector measured, including "flat" / "small" for the bottoms of the scale:
# the operator's rule is that a barely-seen bust should read flat or small, not
# be described by nothing. `BACK_BUST_MIN` is the opt-in exception -- set it to
# e.g. 0.5 to drop the clause on backs below that value instead (it was shipped
# at 0.5 for one round; 0878 0.72 and 0941 0.52 kept their wording then too).
# The low-bust *tags and LoRA* stay off for every back view either way, because
# those assert a chest ("pettanko, tiny breasts") rather than describe one.
BACK_BUST_MIN = 0.0


def load_views():
    if not os.path.exists(VIEWS):
        return {}
    try:
        with open(VIEWS, encoding="utf-8") as fh:
            return json.load(fh)
    except ValueError:
        return {}


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
           equipment_protect=True,
           denoise=1.0, prompt_extra="", max_mp=1.5, sam_refine=2,
           lora=None, lora_strength=1.0, probe_merge=False, inplace_patch=False,
           base_prompt=None):
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
        str(nid + 1): {"class_type": "AmodalSafeFrame", "inputs": {
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
        str(nid + 6): {"class_type": "AmodalMaskUnionFill", "inputs": {
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
        str(nid + 13): {"class_type": "AmodalMaskUnionFill", "inputs": {
            "mask1": [str(nid + 10), 0], "mask2": [str(nid + 11), 0], "mask3": [str(nid + 12), 0],
            "grow": 4, "min_area": 100}},
        str(nid + 27): {"class_type": "AmodalParseClothes", "inputs": {
            "image": [str(nid + 1), 0], "grow": 6}},
        str(nid + 28): {"class_type": "MaskComposite", "inputs": {
            "destination": [str(nid + 6), 0], "source": [str(nid + 27), 0],
            "x": 0, "y": 0, "operation": "add"}},
        str(nid + 14): {"class_type": "MaskComposite", "inputs": {
            "destination": [str(nid + 28), 0], "source": [str(nid + 13), 0],
            "x": 0, "y": 0, "operation": "subtract"}},
        str(nid + 16): ({"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["904", 0], "prompt": (base_prompt or PROMPT) + prompt_extra, "vae": ["902", 0],
            "image1": [str(nid + 1), 0]}} if legacy else
            {"class_type": "AmodalTextEncode", "inputs": {
            "clip": ["904", 0], "prompt": (base_prompt or PROMPT) + prompt_extra, "vae": ["902", 0],
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
        str(nid + 22): {"class_type": "AmodalFinalize", "inputs": {
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
    # The controlnet patch. `inplace_patch` keeps the same MODEL object so
    # ComfyUI does not treat each image as a new model and reload the unet.
    n[str(nid + 15)] = {
        "class_type": ("AmodalControlnetInPlace" if inplace_patch
                       else "QwenImageDiffsynthControlnet"),
        "inputs": {
            "model": model_ref, "model_patch": ["903", 0], "vae": ["902", 0],
            "image": [str(nid + 1), 0], "mask": [str(nid + 14), 0],
            "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}}
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
    if equipment_protect:
        # Props are not clothing.  Anything the mask swallows here is painted
        # over with skin, and a rig or a held weapon usually has no anatomy to
        # fall back on (0313: white smear plus a missing arm).
        n[str(nid + 35)] = {"class_type": "CLIPTextEncode",
                            "inputs": {"text": EQUIPMENT, "clip": ["900", 1]}}
        n[str(nid + 36)] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["900", 0], "image": [str(nid + 1), 0],
            "conditioning": [str(nid + 35), 0],
            "threshold": protect_threshold, "refine_iterations": sam_refine,
            "individual_masks": False}}
        prot_ids.append(nid + 36)
    ins = {"mask%d" % (i + 1): [str(x), 0] for i, x in enumerate(prot_ids)}
    ins.update({"grow": 4, "min_area": 100})
    n[str(nid + 13)]["inputs"] = ins

    if probe_merge:
        # Fold the five detector nodes into one AmodalDetectMany, which runs the
        # vision backbone once and then detects per prompt.  The text encoders
        # stay: only the forward pass is shared.  Output order below must match
        # the conditioning order handed to the node.
        conds = [nid + 2, nid + 3, nid + 7, nid + 9]        # outfit, garments, face, hand
        order = {"outfit": 0, "garments": 1, "face": 2, "hand": 3}
        k = 4
        if footwear_protect:
            conds.append(nid + 33)
            order["footwear"] = k
            k += 1
        if equipment_protect:
            conds.append(nid + 35)
            order["equipment"] = k
            k += 1
        if hair_protect:
            conds.append(nid + 8)
            order["hair"] = k
        det = {"model": ["900", 0], "image": [str(nid + 1), 0],
               "threshold": 0.30, "protect_threshold": protect_threshold,
               "refine_iterations": sam_refine,
               "clothing_prompts": 2}
        for i, c in enumerate(conds, 1):
            det["conditioning%d" % i] = [str(c), 0]
        n[str(nid + 4)] = {"class_type": "AmodalDetectMany", "inputs": det}
        for j in (nid + 5, nid + 10, nid + 11, nid + 12, nid + 34, nid + 36):
            n.pop(str(j), None)
        n[str(nid + 6)]["inputs"]["mask1"] = [str(nid + 4), order["outfit"]]
        n[str(nid + 6)]["inputs"]["mask2"] = [str(nid + 4), order["garments"]]
        prot = ["face", "hand"]
        if footwear_protect:
            prot.append("footwear")
        if equipment_protect:
            prot.append("equipment")
        if hair_protect:
            prot.append("hair")
        n[str(nid + 13)]["inputs"] = {"mask%d" % (i + 1): [str(nid + 4), order[p]]
                                      for i, p in enumerate(prot)}
        n[str(nid + 13)]["inputs"].update({"grow": 4, "min_area": 100})
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


def dedupe_lora(wf, nids):
    """One LoraLoaderModelOnly per distinct (lora, strength) in the chunk.

    `LoraLoaderModelOnly` clones the ModelPatcher, and ComfyUI compares loaded
    models by patcher identity, so a per-image LoRA node makes every image a
    "different" model and reloads the 9.4 GB unet. The gated flat-chest LoRA
    only takes two values per chunk (on or off), so the node is shared and at
    most two variants exist — two loads instead of one per image (ADR-0014).
    """
    seen = {}
    for nid in nids:
        key = str(nid + 40)
        node = wf.get(key)
        if not node or node.get("class_type") != "LoraLoaderModelOnly":
            continue
        sig = (node["inputs"]["lora_name"], node["inputs"]["strength_model"])
        if sig in seen:
            wf[str(nid + 15)]["inputs"]["model"] = [seen[sig], 0]
            del wf[key]
        else:
            seen[sig] = key


def chain_inplace_patches(wf, nids):
    """Sequence in-place controlnet patches against the previous image's sample.

    `AmodalControlnetInPlace` mutates the shared MODEL rather than returning a
    clone, so image i's patch must be applied after image i-1 has finished
    sampling. Without this the last patch would be the only one any sampler
    ever saw. This is a correctness constraint, not a speed one, so it applies
    even when `--no-stage-order` is set.
    """
    for k, nid in enumerate(nids):
        node = wf.get(str(nid + 15), {})
        if node.get("class_type") == "AmodalControlnetInPlace" and k > 0:
            node["inputs"]["dependency"] = [str(nids[k - 1] + 19), 0]


def stage_order(wf, nids):
    """Force stage-major execution across the images of one chunk.

    ComfyUI schedules depth-first from the output node, so a chunk renders image
    1 end to end before image 2 starts, and chunking therefore amortises nothing:
    every image reloads the text encoder (8.7 GB) and the unet (9.4 GB), which
    on a 12 GB card cannot be co-resident anyway (measured in ADR-0013).

    Two identity gates fix the order without touching any value. Every sampler's
    latent and mask are routed through a gate that depends on every image's
    encoding and every image's detection, so nothing samples until every
    detection and every encoding is done. The chunk then runs detect-all,
    encode-all, sample-all, and each model loads once instead of once per image.

    Both gates return their input object unchanged, so outputs stay
    bit-identical to the ungated graph.
    """
    for k, nid in enumerate(nids):
        gm, gl = str(9000 + 2 * k), str(9001 + 2 * k)
        wf[gm] = {"class_type": "AmodalStageGateMask",
                  "inputs": {"value": [str(nid + 14), 0]}}
        wf[gl] = {"class_type": "AmodalStageGateLatent",
                  "inputs": {"value": [str(nid + 17), 0]}}
        for j, other in enumerate(nids, 1):
            wf[gm]["inputs"]["dependency%d" % j] = [str(other + 14), 0]
            wf[gl]["inputs"]["dependency%d" % j] = [str(other + 16), 0]
        wf[str(nid + 18)]["inputs"]["samples"] = [gl, 0]
        wf[str(nid + 18)]["inputs"]["mask"] = [gm, 0]

        # Third barrier: no image decodes until every image has sampled, or the
        # decode + BiRefNet of image 1 evicts the unet before image 2 samples
        # (measured: unet reloaded 3x for a 3-image chunk even after the above).
        gt = str(9100 + k)
        wf[gt] = {"class_type": "AmodalStageGateLatent",
                  "inputs": {"value": [str(nid + 19), 0]}}
        for j, other in enumerate(nids, 1):
            wf[gt]["inputs"]["latent_dependency%d" % j] = [str(other + 19), 0]
        wf[str(nid + 20)]["inputs"]["samples"] = [gt, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
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
    ap.add_argument("--no-equipment-protect", action="store_true",
                    help="stop protecting weapons, bags and worn gear from the clothing "
                         "mask (they come out painted over, see 0313)")
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="sampler denoise inside the clothing mask (1.0 = repaint fully)")
    ap.add_argument("--view-gate", action="store_true",
                    help="route sources that output/axes/_views.json calls 'back' "
                         "(tagger AND Qwen3-VL-4B agreement, see view_axis.py) down "
                         "the back branch")
    ap.add_argument("--assume-back", nargs="*", default=None,
                    help="stems to render with the back branch regardless of the gate "
                         "(substring match), e.g. 3/4 backs the tagger reads as front")
    ap.add_argument("--back-denoise", type=float, default=1.0,
                    help="denoise for back views; at 4 steps anything above 0.80 is a "
                         "no-op and 0.80 keeps sleeve shading (ADR-0008), A/B only")
    ap.add_argument("--no-back-prompt", action="store_true",
                    help="back views: keep the front prompt (A/B)")
    ap.add_argument("--no-back-low", action="store_true",
                    help="back views: keep the low-bust tags/LoRA (A/B)")
    ap.add_argument("--prompt-extra", default="",
                    help="extra sentence appended to the body prompt (trial only)")
    ap.add_argument("--prompt-base", default=None,
                    help="replace the base prompt for every image (A/B only)")
    ap.add_argument("--no-profile", action="store_true",
                    help="do not append the per-character body profile to the prompt")
    ap.add_argument("--axes", action="store_true",
                    help="append the VLM axis fragment from output/axes/<stem>.json")
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
    ap.add_argument("--no-probe-merge", action="store_true",
                    help="run the five SAM3 probes as five separate nodes (A/B)")
    ap.add_argument("--no-stage-order", action="store_true",
                    help="schedule image-by-image instead of stage-by-stage (A/B)")
    ap.add_argument("--no-inplace-patch", action="store_true",
                    help="clone the model for the controlnet patch (stock node, A/B)")
    ap.add_argument("--group-lora", action="store_true",
                    help="sort LoRA variants together so chunks stay homogeneous "
                         "(measured neutral on the current corpus, ADR-0014)")
    ap.add_argument("--probe-merge", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sync", action="store_true",
                    help="regenerate the UI workflow at startup (default: skip)")
    ap.add_argument("--no-sync", action="store_true", help=argparse.SUPPRESS)
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
            "\"--use-ck-attention\",\n"
            "                  \"--input-directory\",\"$PWD\\input\","
            "\"--output-directory\",\"$PWD\\output\",\n"
            "                  \"--extra-model-paths-config\",\"$PWD\\comfy_nodes.yaml\" `\n"
            "    -RedirectStandardOutput \"..\\server.log\" "
            "-RedirectStandardError \"..\\server.err.log\"\n\n"
            "Check with:\n"
            "  Invoke-WebRequest http://127.0.0.1:8199/system_stats -UseBasicParsing | "
            "Select-Object StatusCode" % t.HOST)

    if a.sync:
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
    assume = a.assume_back or ()
    views = load_views() if a.view_gate else {}
    back_stems = [stem for _, stem in todo if is_back(stem, views, assume)]
    if back_stems:
        print("back branch: %d image(s)" % len(back_stems))
        for s in back_stems:
            how = ("listed" if any(x.lower() in s.lower() for x in assume)
                   else views.get(s, {}).get("answer") or "gate")
            print("    %-40s %s" % (s, how))
    low = []
    if a.axes and not a.no_low_lora and a.lora is None:
        back_set = set(back_stems)
        for rel, stem in todo:
            p = os.path.join(C.AXES, stem + ".json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                b = json.load(fh).get("bust_measured")
            if b is not None and b < LOW_LORA_BELOW and stem not in back_set:
                low.append(stem)
    if low:
        try:
            import urllib.request as _u
            info = json.loads(_u.urlopen(
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

    # Sort by LoRA variant before chunking. The flat-chest LoRA fires per image
    # on bust < LOW_LORA_BELOW (known from output/axes/*.json, long before any
    # node runs), and each variant is a distinct model to ComfyUI, so an unsorted
    # chunk reloads the 9.4 GB unet every time the variant alternates.
    # Chunking stays by size: splitting at the variant boundary would add a whole
    # extra chunk (one more text-encoder + SAM3 load) for no gain (ADR-0014).
    if a.group_lora and low:
        low_set = set(low)
        todo.sort(key=lambda t: 1 if t[1] in low_set else 0)
    chunks = [todo[ci:ci + a.chunk] for ci in range(0, len(todo), a.chunk)]

    for ci, chunk in enumerate(chunks):
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
            back = is_back(stem, views, assume)
            base = a.prompt_base or PROMPT
            denoise = a.denoise
            if back and not a.prompt_base:
                if not a.no_back_prompt:
                    base = BACK_PROMPT
                if a.back_denoise != 1.0:
                    denoise = a.back_denoise
                print("    back branch: %s%s%s"
                      % ("front prompt kept" if a.no_back_prompt else "back prompt",
                         ", low-bust tags kept" if a.no_back_low else ", no low-bust",
                         "" if a.back_denoise == 1.0 else
                         ", denoise %.2f" % a.back_denoise))
            if not a.no_profile:
                try:
                    import image_profile
                    extra = extra + image_profile.profile_for(
                        os.path.join(INPUT, rel))["prompt"]
                except Exception as e:
                    print("    profile skipped:", e)
            if a.axes:
                apath = os.path.join(C.AXES, stem + ".json")
                if os.path.exists(apath):
                    with open(apath, encoding="utf-8") as fh:
                        rec = json.load(fh)
                    frag = rec.get("parts_cal") or rec.get("parts") or []
                    style = rec.get("style") or []
                    bust = rec.get("bust_measured")
                    if back and (bust is None or bust < BACK_BUST_MIN):
                        # Only when the operator opts in via BACK_BUST_MIN, or when
                        # there is no measurement at all: then there is nothing to
                        # describe.
                        chest = set(rec.get("chest") or [])
                        frag = [p for p in frag if p not in chest]
                    low_ok = not (back and not a.no_back_low)
                    if bust is not None and bust < LOW_BUST_BELOW and low_ok:
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
                                    not a.no_footwear_protect,
                                    not a.no_equipment_protect, denoise, extra,
                                    a.max_mp, a.sam_refine, lora, lora_strength,
                                    not a.no_probe_merge, not a.no_inplace_patch, base)
            wf.update(nodes)
            saves[stem] = save_id
            nid += 100

        if len(chunk) > 1 and not a.no_stage_order:
            stage_order(wf, list(range(1000, 1000 + 100 * len(chunk), 100)))
        chain_inplace_patches(wf, list(range(1000, 1000 + 100 * len(chunk), 100)))
        if not a.no_inplace_patch:
            dedupe_lora(wf, list(range(1000, 1000 + 100 * len(chunk), 100)))

        tc = time.time()
        r = t.post_prompt(wf)
        print(f"[chunk {ci+1}/{len(chunks)}] {len(chunk)} images, submitted {r['prompt_id']}", r.get("node_errors"))
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
