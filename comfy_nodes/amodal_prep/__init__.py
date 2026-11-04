"""Safe-frame prep, mask hole-filling and finalise-back nodes."""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import binary_fill_holes, binary_dilation, label


# Stage timings for `AmodalDetectMany`. Off by default: the synchronisation it
# needs would add a stall to every call, and the numbers belong in the ADR, not
# in a production log. Set AMODAL_TIMING=1 before starting the server.
_TIMING = os.environ.get("AMODAL_TIMING") == "1"


def _to_pil(t):
    a = (t.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, "L")
    return Image.fromarray(a[:, :, :3], "RGB")


def _to_tensor(im):
    a = np.array(im).astype(np.float32) / 255.0
    if a.ndim == 2:
        return torch.from_numpy(a)[None, :, :]
    return torch.from_numpy(a)[None, :, :, :3]


class SafeFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "margin": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
            "scale": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 4.0, "step": 0.05}),
            "max_megapixels": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
            "min_aspect": ("FLOAT", {"default": 0.62, "min": 0.2, "max": 1.0, "step": 0.01}),
            "max_aspect": ("FLOAT", {"default": 1.60, "min": 1.0, "max": 4.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE", "FRAME")
    RETURN_NAMES = ("canvas", "frame")
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, image, mask, margin, scale, max_megapixels, min_aspect, max_aspect):
        rgb = _to_pil(image[0])
        W, H = rgb.size
        m = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
        alpha = np.clip(1.0 - m, 0.0, 1.0)

        src = np.array(rgb).astype(np.float32)
        a3 = alpha[:, :, None]
        comp = src * a3 + 255.0 * (1.0 - a3)

        mm = int(round(margin * max(W, H)))
        Wp, Hp = W + 2 * mm, H + 2 * mm
        asp = Wp / Hp
        if asp < min_aspect:
            Wp = int(round(Hp * min_aspect))
        elif asp > max_aspect:
            Hp = int(round(Wp / max_aspect))
        ox, oy = (Wp - W) // 2, (Hp - H) // 2

        s = scale
        if max_megapixels > 0 and (Wp * s) * (Hp * s) > max_megapixels * 1e6:
            s = float(np.sqrt(max_megapixels * 1e6 / (Wp * Hp)))
        Wr = max(16, int(round(Wp * s / 16)) * 16)
        Hr = max(16, int(round(Hp * s / 16)) * 16)

        canvas = np.full((Hp, Wp, 3), 255.0, np.float32)
        canvas[oy:oy + H, ox:ox + W] = comp
        out = Image.fromarray(canvas.round().astype(np.uint8), "RGB").resize((Wr, Hr), Image.LANCZOS)

        frame = {"W": W, "H": H, "Wp": Wp, "Hp": Hp, "ox": ox, "oy": oy, "Wr": Wr, "Hr": Hr}
        return (_to_tensor(out), frame)


class MaskUnionFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask1": ("MASK",),
            "grow": ("INT", {"default": 16, "min": 0, "max": 256}),
            "min_area": ("INT", {"default": 400, "min": 0, "max": 100000}),
        }, "optional": {
            "mask2": ("MASK",),
            "mask3": ("MASK",),
            "mask4": ("MASK",),
            "mask5": ("MASK",),
            "mask6": ("MASK",),
            "mask7": ("MASK",),
            "mask8": ("MASK",),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, mask1, grow, min_area, mask2=None, mask3=None, mask4=None,
            mask5=None, mask6=None, mask7=None, mask8=None):
        ms = [m[0].cpu().numpy() > 0.5 for m in
              (mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8)
              if m is not None]
        union = np.logical_or.reduce(ms)

        lab, n = label(union)
        filled = np.zeros_like(union)
        for i in range(1, n + 1):
            comp = lab == i
            if comp.sum() >= min_area:
                filled |= binary_fill_holes(comp)

        if grow > 0:
            filled = binary_dilation(filled, iterations=grow)

        return (torch.from_numpy(filled.astype(np.float32))[None, :, :],)


class FinalizeAmodal:
    """Map the render back to source size, preserving the source's soft alpha.

    Two rules:
      * alpha stays float - no binary dilation, so anti-aliased hair survives
      * in the semi-transparent edge band, the SOURCE pixel and alpha are used
        verbatim, because the render there is contaminated by the canvas colour
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "frame": ("FRAME",),
            "source_image": ("IMAGE",),
            "source_mask": ("MASK",),
            "edge_soften": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 4.0, "step": 0.1}),
            "soft_alpha_below": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 1.0, "step": 0.01}),
            "bg_tolerance": ("FLOAT", {"default": 18.0, "min": 2.0, "max": 80.0, "step": 1.0,
                                       "tooltip": "How far a render pixel must differ from the canvas colour to count as body outside the source silhouette."}),
        }, "optional": {
            "edited_mask": ("MASK",),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, image, mask, frame, source_image, source_mask, edge_soften, soft_alpha_below, bg_tolerance, edited_mask=None):
        t0 = time.perf_counter() if _TIMING else 0.0
        W, H = frame["W"], frame["H"]
        Wp, Hp = frame["Wp"], frame["Hp"]
        ox, oy = frame["ox"], frame["oy"]

        ren = np.array(_to_pil(image[0]).resize((Wp, Hp), Image.LANCZOS)
                       .crop((ox, oy, ox + W, oy + H))).astype(np.float32)
        t1 = time.perf_counter() if _TIMING else 0.0

        subj = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
        subj_im = Image.fromarray((np.clip(subj, 0, 1) * 255).astype(np.uint8), "L")
        subj_c = np.array(subj_im.resize((Wp, Hp), Image.LANCZOS)
                          .crop((ox, oy, ox + W, oy + H))).astype(np.float32) / 255.0
        t2 = time.perf_counter() if _TIMING else 0.0

        src_rgb = np.array(_to_pil(source_image[0])).astype(np.float32)
        sm = source_mask[0].cpu().numpy() if source_mask.ndim == 3 else source_mask.cpu().numpy()
        src_a = np.clip(1.0 - sm, 0.0, 1.0)

        if edge_soften > 0:
            a_im = Image.fromarray((src_a * 255).astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(edge_soften))
            a_soft = np.array(a_im).astype(np.float32) / 255.0
        else:
            a_soft = src_a

        # body may be redrawn slightly larger than the original outline, so allow
        # it to extend outside the silhouette wherever the render is clearly not canvas
        border = np.concatenate([ren[0], ren[-1], ren[:, 0], ren[:, -1]])
        bg = np.median(border, axis=0)
        dist = np.abs(ren - bg).max(axis=2)
        lo = bg_tolerance * 0.4
        colourness = np.clip((dist - lo) / max(bg_tolerance - lo, 1e-3), 0.0, 1.0)

        allowance = np.maximum(np.minimum(a_soft, 1.0), colourness)
        matte = np.clip(subj_c * allowance, 0.0, 1.0)
        t3 = time.perf_counter() if _TIMING else 0.0

        band = src_a < soft_alpha_below
        if edited_mask is not None:
            em = edited_mask[0].cpu().numpy() if edited_mask.ndim == 3 else edited_mask.cpu().numpy()
            em_im = Image.fromarray((np.clip(em, 0, 1) * 255).astype(np.uint8), "L")
            em_c = np.array(em_im.resize((Wp, Hp), Image.LANCZOS)
                            .crop((ox, oy, ox + W, oy + H))).astype(np.float32) / 255.0
            band = band & (em_c < 0.5)   # never restore soft pixels the model repainted
        out_a = np.where(band, src_a, matte)
        out_rgb = np.where(band[:, :, None], src_rgb, ren)

        t4 = time.perf_counter() if _TIMING else 0.0
        res = (_to_tensor(Image.fromarray(np.clip(out_rgb, 0, 255).astype(np.uint8), "RGB")),
               torch.from_numpy((1.0 - out_a).astype(np.float32))[None, :, :])
        if _TIMING:
            t5 = time.perf_counter()
            print("[AmodalFinalize] total %.3fs  render-resize %.3fs  mask-resize %.3fs  "
                  "math+blur %.3fs  edited-mask %.3fs  writeback %.3fs"
                  % (t5 - t0, t1 - t0, t2 - t1, t3 - t2,
                     (t4 - t3) if edited_mask is not None else 0.0, t5 - t4),
                  flush=True)
        return res


class FolderList:
    """List image files in a folder, sorted, as a list for Start Loop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": "."}),
            "pattern": ("STRING", {"default": "*.png"}),
        }}

    RETURN_TYPES = ("FILELIST", "INT")
    RETURN_NAMES = ("files", "count")
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, folder, pattern):
        import glob
        import os
        base = folder
        if not os.path.isabs(base):
            import folder_paths
            base = os.path.join(folder_paths.get_input_directory(), folder)
        files = sorted(glob.glob(os.path.join(base, pattern)))
        files = [f for f in files if os.path.isfile(f)]
        return (files, len(files))


class LoadImagePath:
    """Load one image by path, returning IMAGE + MASK (1 = transparent)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, path):
        import os
        im = Image.open(path).convert("RGBA")
        a = np.array(im).astype(np.float32) / 255.0
        rgb = torch.from_numpy(a[:, :, :3])[None, :, :, :]
        mask = torch.from_numpy(1.0 - a[:, :, 3])[None, :, :]
        return (rgb, mask, os.path.splitext(os.path.basename(path))[0])


class TextEncodeExact:
    """TextEncodeQwenImageEditPlus, but the VAE reference matches the sampling latent.

    The stock node rescales every reference image to roughly 1 megapixel before
    VAE-encoding it, so the reference latent grid is NOT the grid being denoised:
    a 0.74 MP canvas is upscaled 1.19x, a 1.16 MP canvas is downscaled 0.95x.
    The model then reads the body at the wrong scale, which shows up as a few
    pixels of drift, a slight zoom, re-framing and occasionally a lost limb.

    Here the reference is resized to the target latent's pixel size, so both
    grids agree exactly and the edit is positionally locked to the canvas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "vae": ("VAE",),
            "latent": ("LATENT",),
        }, "optional": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    def run(self, clip, prompt, vae, latent, image1=None, image2=None, image3=None):
        import math
        import node_helpers
        import comfy.utils

        llama_template = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
            "size, texture, objects, background), then explain how the user's text instruction "
            "should alter or modify the image. Generate a new image that meets the user's "
            "requirements while maintaining consistency with the original input where "
            "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")

        tw = latent["samples"].shape[-1] * 8
        th = latent["samples"].shape[-2] * 8

        ref_latents = []
        images_vl = []
        image_prompt = ""
        for image in (image1, image2, image3):
            if image is None:
                continue
            samples = image.movedim(-1, 1)

            # vision tower sees a small thumbnail, same rule as the stock node
            vb = math.sqrt(384 * 384 / (samples.shape[3] * samples.shape[2]))
            vs = comfy.utils.common_upscale(
                samples, round(samples.shape[3] * vb), round(samples.shape[2] * vb),
                "area", "disabled")
            images_vl.append(vs.movedim(1, -1))

            # vae reference: exactly the latent's pixel size
            s = comfy.utils.common_upscale(samples, tw, th, "lanczos", "center")
            ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
                len(images_vl))

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl,
                               llama_template=llama_template)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            cond = node_helpers.conditioning_set_values(
                cond, {"reference_latents": ref_latents}, append=True)
        return (cond,)


class ParseClothes:
    """Clothing mask from a fashion-parsing SegFormer, as a fallback for SAM3.

    SAM3 is open-vocabulary and misses things: on a polo shirt it returned the
    body but not the sleeves, and on greaves it returned nothing at all, so both
    stayed in the output. A parsing model is trained per pixel on clothing
    classes instead of matching a phrase, so sleeves belong to "upper-clothes"
    and greaves belong to "shoe" by construction.

    Returns a mask of everything that should be REMOVED (upper-clothes, skirt,
    pants, dress, belt), with hat / shoes / bag / scarf / face / hair subtracted.
    """

    MODEL = "mattmdjaga/segformer_b2_clothes"
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "grow": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "image/amodal"

    @classmethod
    def _load(cls):
        if "m" not in cls._cache:
            import torch
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
            proc = SegformerImageProcessor.from_pretrained(cls.MODEL)
            model = SegformerForSemanticSegmentation.from_pretrained(cls.MODEL).eval()
            if torch.cuda.is_available():
                model = model.cuda()
            cls._cache["m"] = (proc, model)
        return cls._cache["m"]

    def run(self, image, grow):
        import torch
        import torch.nn.functional as F

        proc, model = self._load()
        dev = next(model.parameters()).device

        rgb = (image[0, :, :, :3].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(rgb, "RGB")

        inputs = proc(images=pil, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = F.interpolate(logits, size=pil.size[::-1], mode="bilinear",
                               align_corners=False)
        lab = logits.argmax(1)[0].cpu().numpy()

        # 4 upper-clothes, 5 skirt, 6 pants, 7 dress, 8 belt
        clothing = np.isin(lab, [4, 5, 6, 7, 8])
        # 1 hat, 9/10 shoes, 16 bag, 17 scarf, 11 face, 2 hair, 3 sunglasses
        keep = np.isin(lab, [1, 2, 3, 9, 10, 11, 16, 17])
        mask = clothing & ~keep

        if grow > 0:
            mask = binary_dilation(mask, iterations=int(grow))

        return (torch.from_numpy(mask.astype(np.float32))[None, :, :],)


def _tracker_features(sam3_model, crop_frame):
    """Tracker features for one crop, with the ViTDet trunk run once.

    Mirrors the head of `SAM3Model.forward_segment` (detector.py:535-554) but
    takes the trunk output as an argument, so a caller that refines the same
    crop twice pays for the trunk once.
    """
    from comfy.ops import cast_to_input

    detector = sam3_model.detector
    bb = detector.backbone["vision_backbone"]
    trunk = bb.trunk(crop_frame)
    _, _, features, _ = bb(crop_frame, tracker_mode="interactive",
                           cached_trunk=trunk, tracker_only=True)
    if not bb.multiplex and detector.scalp > 0:
        features = features[:-detector.scalp]
    high_res = list(features[:-1])
    backbone_feat = features[-1]
    B, C, Hf, Wf = backbone_feat.shape
    no_mem = getattr(sam3_model.tracker, "interactivity_no_mem_embed", None)
    if no_mem is None:
        no_mem = getattr(sam3_model.tracker, "no_mem_embed", None)
    if no_mem is not None:
        flat = backbone_feat.flatten(2).permute(0, 2, 1)
        flat = flat + cast_to_input(no_mem, flat)
        backbone_feat = flat.view(B, Hf, Wf, C).permute(0, 3, 1, 2)
    return high_res, backbone_feat


def _refine_cached(sam3_model, orig_image_hwc, coarse_mask, box_xyxy,
                   H, W, device, dtype, iterations):
    """`_refine_mask` without the repeated backbone pass over the same crop.

    `_refine_mask` (nodes_sam3.py:76-78) calls `forward_segment` once per
    iteration, and `forward_segment` starts by re-running the visual backbone
    on the crop (detector.py:537). The crop does not change between iterations,
    so all but the first pass are identical recomputation: measured at 3.6 s per
    image across 27 objects at iterations=2, which was half the detection
    stage. Hoisting the features out of the loop is a pure win because it
    changes no operand, and the dumped masks come back bit-identical.
    """
    import comfy.utils

    def fallback():
        return (F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False)[0] > 0).float()

    if iterations <= 0:
        return fallback()

    pad_frac = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))
    if cx2 <= cx1 or cy2 <= cy1:
        return fallback()

    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_1008 = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1),
                                          1008, 1008, "bilinear", crop="disabled")
    crop_frame = crop_1008.to(device=device, dtype=dtype)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1

    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    if mx2 <= mx1 or my2 <= my1:
        return fallback()

    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    high_res, backbone_feat = _tracker_features(sam3_model, crop_frame)
    for _ in range(iterations):
        coarse_input = F.interpolate(mask_logit, size=(1008, 1008),
                                     mode="bilinear", align_corners=False)
        _, mask_logit, _, _ = sam3_model.tracker._forward_sam_heads(
            backbone_features=backbone_feat,
            point_inputs=None,
            mask_inputs=coarse_input,
            box_inputs=None,
            high_res_features=high_res,
            multimask_output=False,
        )

    refined_crop = F.interpolate(mask_logit, size=(crop_h, crop_w),
                                 mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                                mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()


class StageGateLatent:
    """Ordering-only node: returns `value` unchanged once every dependency exists.

    ComfyUI executes depth-first from the output node, so a chunk renders image 1
    end to end before starting image 2. On a 12 GB card the text encoder (8.7 GB)
    and the unet (9.4 GB) cannot be co-resident, so that ordering reloads the
    encoder once per image.

    Wiring every sampler's latent through one gate that depends on every
    encoding forces all encodings first, then all samplers, so each model loads
    once per chunk. No arithmetic happens here: the same tensor object is
    returned, so the result is bit-identical.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("LATENT",)},
                "optional": dict(
                    {"dependency%d" % i: ("CONDITIONING",) for i in range(1, 33)},
                    **{"latent_dependency%d" % i: ("LATENT",) for i in range(1, 33)})}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "amodal_prep"

    def run(self, value, **kwargs):
        return (value,)


class StageGateMask:
    """Ordering-only node for masks; same reasoning as `StageGateLatent`.

    Forces every image's detection to finish before any image's mask is used,
    so SAM3 loads once per chunk instead of once per image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("MASK",)},
                "optional": {"dependency%d" % i: ("MASK",) for i in range(1, 33)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "amodal_prep"

    def run(self, value, **kwargs):
        return (value,)


class ControlnetInPlace:
    """Apply the Qwen diffsynth controlnet to the model without cloning it.

    The stock node (`QwenImageDiffsynthControlnet`) does `model.clone()` and
    registers the patch on the clone. ComfyUI identifies a loaded model by
    ModelPatcher identity — `LoadedModel.__eq__` is `self.model is other.model`
    — so each image presents a "different" model and the 9.4 GB unet is unloaded
    and reloaded between images. Measured at 6-37 s per image, which on a
    100-image batch is the single largest remaining cost (ADR-0014).

    Patching the same object in place keeps the identity, so the weights stay
    resident. `set_model_patch` appends to a list, so the entry is REPLACED here;
    appending would leave every earlier image's controlnet active and silently
    stack the masks.

    The caller must guarantee this node runs immediately before the sampler that
    consumes it, because the patch lives on a shared object: the returned MODEL
    is the same one received. `run_folder.stage_order` chains every patch to the
    previous image's sampler to enforce that.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "model_patch": ("MODEL_PATCH",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        }, "optional": {
            "mask": ("MASK",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "dependency": ("LATENT",),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "amodal_prep"

    def apply(self, model, model_patch, vae, image, strength=1.0, mask=None,
              start_percent=0.0, end_percent=1.0, dependency=None):
        from comfy_extras.nodes_model_patch import DiffSynthCnetPatch

        if strength == 0 or image is None:
            return (model,)

        sampling = model.get_model_object("model_sampling")
        sigma_range = (float(sampling.percent_to_sigma(start_percent)),
                       float(sampling.percent_to_sigma(end_percent)))
        image = image[:, :, :, :3]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = 1.0 - mask

        patch = DiffSynthCnetPatch(model_patch, vae, image, strength, mask,
                                   sigma_range=sigma_range)
        to = model.model_options.setdefault("transformer_options", {})
        patches = to.setdefault("patches", {})
        patches["double_block"] = [patch]      # replace, never append
        return (model,)


class DetectAmodal:
    """Several SAM3 text prompts over one image, with a single backbone pass.

    `SAM3_Detect` loops over the prompts it is given and calls the detector once
    per prompt (comfy_extras/nodes_sam3.py:208), and `SAM3Detector.forward`
    re-runs the ViTDet trunk plus the FPN on the same image every call
    (detector.py:473).  The pipeline asks five questions of every image, so the
    backbone currently runs five times on identical pixels.

    The two halves are separable in the installed code: the backbone accepts a
    cached trunk (sam.py:396) and `_detect()` only needs the feature pyramid.
    This node computes the features once and then runs detection per prompt.

    Boxes and scores come back normalised and masks at detector resolution, so
    both are rescaled here exactly as `SAM3Detect.forward` does via `orig_size`.
    Keeping that identical is what lets the merged path be compared against the
    five-node path without a quality difference appearing from the rescaling.

    The pipeline runs the clothing prompts at 0.30 and the protection prompts at
    `--protect-threshold` (0.25 by default), so a single threshold would silently
    change the masks. `clothing_prompts` names how many leading conditionings
    take `threshold`; the rest take `protect_threshold`.

    Refinement runs through `_refine_cached` rather than `_refine_mask`, which
    removes a redundant backbone pass per object and no pixels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "conditioning1": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0,
                                        "step": 0.01}),
                "protect_threshold": ("FLOAT", {"default": 0.25, "min": 0.0,
                                                "max": 1.0, "step": 0.01}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5}),
                "clothing_prompts": ("INT", {"default": 2, "min": 0, "max": 6}),
            },
            "optional": {"conditioning%d" % i: ("CONDITIONING",) for i in range(2, 9)},
        }

    RETURN_TYPES = ("MASK",) * 8
    RETURN_NAMES = ("mask1", "mask2", "mask3", "mask4", "mask5", "mask6",
                    "mask7", "mask8")
    FUNCTION = "run"
    CATEGORY = "amodal_prep"

    def run(self, model, image, conditioning1, threshold=0.30,
            protect_threshold=0.25, refine_iterations=2, clothing_prompts=2,
            conditioning2=None, conditioning3=None,
            conditioning4=None, conditioning5=None, conditioning6=None,
            conditioning7=None, conditioning8=None):
        import torch.nn.functional as F
        import comfy.model_management
        import comfy.utils
        from comfy_extras.nodes_sam3 import _extract_text_prompts

        conds = [c for c in (conditioning1, conditioning2, conditioning3,
                             conditioning4, conditioning5, conditioning6,
                             conditioning7, conditioning8) if c]

        B, H, W, C = image.shape
        image_in = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        detector = sam3_model.detector
        resizer = detector.backbone["language_backbone"]["resizer"]

        out = [[] for _ in conds]
        with torch.no_grad():
            for b in range(B):
                frame = image_in[b:b + 1].to(device=device, dtype=dtype)
                # The one call the five-node path repeats five times.
                if _TIMING:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                features, positions, _, _ = detector._get_backbone_features(frame)
                if _TIMING:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                t_ref = 0.0
                n_ref = 0

                for ci, cond in enumerate(conds):
                    found = []
                    # clothing prompts first (0.30), protection after
                    # (--protect-threshold), matching the five-node path.
                    thr = threshold if ci < clothing_prompts else protect_threshold
                    # max_det and the attention mask travel with the conditioning,
                    # same tuple shape SAM3_Detect consumes.
                    for emb, tmask, max_det in _extract_text_prompts(cond, device, dtype):
                        emb = resizer(emb.to(device=device, dtype=dtype))
                        # _detect does not coerce the mask; forward() and
                        # forward_from_trunk() both call .bool(). _extract_text_prompts
                        # hands back int64, which sage-attention rejects.
                        if tmask is not None:
                            tmask = tmask.bool()
                        # _detect returns a 4-tuple; only forward() wraps it in a dict.
                        boxes, scores, masks, _ = detector._detect(
                            features, positions, emb, tmask)
                        # Boxes are normalised cxcywh -> xyxy; rescale to pixels the
                        # way forward() does with orig_size, because _refine_mask
                        # crops with them in pixel space.
                        boxes = boxes[0] * torch.tensor(
                            [W, H, W, H], device=boxes.device, dtype=boxes.dtype)
                        probs = scores[0].sigmoid()
                        masks = masks[0]

                        keep = probs > thr
                        boxes = boxes[keep].cpu()
                        probs = probs[keep].cpu()
                        masks = masks[keep]

                        order = probs.argsort(descending=True)[:max_det]
                        boxes = boxes[order]
                        masks = masks[order]

                        if masks.numel():
                            masks = F.interpolate(
                                masks.unsqueeze(1), size=(H, W),
                                mode="bilinear", align_corners=False)[:, 0]
                        for m, box in zip(masks, boxes):
                            if _TIMING:
                                torch.cuda.synchronize()
                                tr0 = time.perf_counter()
                            found.append(_refine_cached(
                                sam3_model, image[b], m, box, H, W,
                                device, dtype, refine_iterations))
                            if _TIMING:
                                torch.cuda.synchronize()
                                t_ref += time.perf_counter() - tr0
                                n_ref += 1

                    if found:
                        merged = torch.cat(found, dim=0)
                        out[ci].append((merged > 0).any(dim=0).float())
                    else:
                        out[ci].append(torch.zeros(H, W, device=(
                            comfy.model_management.intermediate_device())))

                # The split that decides whether sharing the backbone is worth it:
                # everything after t1 is per-prompt work the merge cannot remove.
                if _TIMING:
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    print("[AmodalDetectMany] backbone %.2fs  detect %.2fs  "
                          "refine %.2fs/%d obj  %d prompts"
                          % (t1 - t0, t2 - t1, t_ref, n_ref, len(conds)),
                          flush=True)

        z = torch.zeros(H, W)
        result = []
        for ci in range(8):
            result.append(torch.stack(out[ci]) if ci < len(out) else z[None])
        return tuple(result)


NODE_CLASS_MAPPINGS = {
    "AmodalDetectMany": DetectAmodal,
    "AmodalSafeFrame": SafeFrame,
    "AmodalTextEncode": TextEncodeExact,
    "AmodalParseClothes": ParseClothes,
    "AmodalMaskUnionFill": MaskUnionFill,
    "AmodalFinalize": FinalizeAmodal,
    "AmodalFolderList": FolderList,
    "AmodalLoadImagePath": LoadImagePath,
    "AmodalStageGateLatent": StageGateLatent,
    "AmodalStageGateMask": StageGateMask,
    "AmodalControlnetInPlace": ControlnetInPlace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AmodalDetectMany": "SAM3 Detect Many (shared backbone)",
    "AmodalSafeFrame": "Safe Frame (amodal)",
    "AmodalTextEncode": "Text Encode Qwen Edit (exact ref size)",
    "AmodalParseClothes": "Parse Clothes (segformer)",
    "AmodalMaskUnionFill": "Mask Union + Fill Holes",
    "AmodalFinalize": "Finalize to Source Size",
    "AmodalFolderList": "Folder List (amodal)",
    "AmodalLoadImagePath": "Load Image By Path",
    "AmodalStageGateLatent": "Stage Gate (latent, ordering only)",
    "AmodalStageGateMask": "Stage Gate (mask, ordering only)",
    "AmodalControlnetInPlace": "Diffsynth ControlNet (in place, no clone)",
}
