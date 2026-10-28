import json
import os
import sys
import time
import uuid
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

HOST = C.HOST
CKPT = "waiNSFWIllustrious_v140.safetensors"

POS_TXT = (
    "1girl, solo, adult, mature female, nude, completely nude, bare breasts, "
    "bare shoulders, collarbone, navel, standing, full body, simple background, "
    "best quality, highres"
)
NEG_TXT = (
    "clothes, shirt, skirt, dress, bra, panties, underwear, pantyhose, socks, shoes, "
    "censored, mosaic, bar, text, watermark, bad anatomy, extra limbs, deformed"
)


def post_prompt(workflow):
    payload = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode("utf-8")
    req = urllib.request.Request(
        HOST + "/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for(prompt_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HOST + "/history/" + prompt_id, timeout=30) as r:
                hist = json.loads(r.read().decode("utf-8"))
        except Exception:
            hist = {}
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(2)
    raise TimeoutError("prompt did not finish: " + prompt_id)


def txt2img():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": POS_TXT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_TXT, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": 123456789,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "test_txt2img"}},
    }
    return wf


def inpaint():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": POS_TXT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_TXT, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "sample3_gray.png"}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": "sample3_mask.png", "channel": "red"}},
        "6": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["4", 0], "vae": ["1", 2], "mask": ["5", 0], "grow_mask_by": 16},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": 123456789,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
                "denoise": 0.9,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "test_inpaint"}},
    }
    return wf


def crop_inpaint():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": POS_TXT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_TXT, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "sample3_gray.png"}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": "sample3_mask.png", "channel": "red"}},
        "6": {
            "class_type": "InpaintCropImproved",
            "inputs": {
                "image": ["4", 0],
                "mask": ["5", 0],
                "downscale_algorithm": "bilinear",
                "upscale_algorithm": "bicubic",
                "preresize": False,
                "preresize_mode": "ensure minimum resolution",
                "preresize_min_width": 1024,
                "preresize_min_height": 1024,
                "preresize_max_width": 4096,
                "preresize_max_height": 4096,
                "mask_fill_holes": True,
                "mask_expand_pixels": 12,
                "mask_invert": False,
                "mask_blend_pixels": 16,
                "mask_hipass_filter": 0.1,
                "extend_for_outpainting": False,
                "extend_up_factor": 1.0,
                "extend_down_factor": 1.0,
                "extend_left_factor": 1.0,
                "extend_right_factor": 1.0,
                "context_from_mask_extend_factor": 1.8,
                "output_resize_to_target_size": True,
                "output_target_width": 1024,
                "output_target_height": 1024,
                "output_padding": "32",
                "device_mode": "gpu (much faster)",
            },
        },
        "7": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["6", 1], "vae": ["1", 2], "mask": ["6", 2], "grow_mask_by": 16},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": 123456789,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["7", 0],
                "denoise": 0.9,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["6", 0], "inpainted_image": ["9", 0]},
        },
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "test_crop_inpaint"}},
    }
    return wf


def full():
    """Real pipeline: RGBA in -> seam-free inpaint -> RGBA out with original alpha."""
    W, H = 429, 1037
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": POS_TXT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_TXT, "clip": ["1", 1]}},
        # LoadImage returns RGB in output 0 and (1 - alpha) in output 1.
        "4": {"class_type": "LoadImage", "inputs": {"image": "sample3.png"}},
        "5": {"class_type": "InvertMask", "inputs": {"mask": ["4", 1]}},
        "6": {"class_type": "EmptyImage", "inputs": {"width": W, "height": H, "batch_size": 1, "color": 8421504}},
        "7": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["6", 0],
                "source": ["4", 0],
                "x": 0, "y": 0,
                "resize_source": False,
                "mask": ["5", 0],
            },
        },
        "8": {"class_type": "LoadImageMask", "inputs": {"image": "sample3_mask.png", "channel": "red"}},
        "9": {
            "class_type": "InpaintCropImproved",
            "inputs": {
                "image": ["7", 0],
                "mask": ["8", 0],
                "downscale_algorithm": "bilinear",
                "upscale_algorithm": "bicubic",
                "preresize": False,
                "preresize_mode": "ensure minimum resolution",
                "preresize_min_width": 1024,
                "preresize_min_height": 1024,
                "preresize_max_width": 4096,
                "preresize_max_height": 4096,
                "mask_fill_holes": True,
                "mask_expand_pixels": 12,
                "mask_invert": False,
                "mask_blend_pixels": 16,
                "mask_hipass_filter": 0.1,
                "extend_for_outpainting": False,
                "extend_up_factor": 1.0,
                "extend_down_factor": 1.0,
                "extend_left_factor": 1.0,
                "extend_right_factor": 1.0,
                "context_from_mask_extend_factor": 1.8,
                "output_resize_to_target_size": True,
                "output_target_width": 1024,
                "output_target_height": 1024,
                "output_padding": "32",
                "device_mode": "gpu (much faster)",
            },
        },
        "10": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["9", 1], "vae": ["1", 2], "mask": ["9", 2], "grow_mask_by": 16},
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": 123456789,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["10", 0],
                "denoise": 0.9,
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "InpaintStitchImproved", "inputs": {"stitcher": ["9", 0], "inpainted_image": ["12", 0]}},
        "14": {"class_type": "MaskToImage", "inputs": {"mask": ["8", 0]}},
        "15": {"class_type": "ImageBlur", "inputs": {"image": ["14", 0], "blur_radius": 8, "sigma": 1.0}},
        "16": {"class_type": "ImageToMask", "inputs": {"image": ["15", 0], "channel": "red"}},
        "17": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["7", 0],
                "source": ["13", 0],
                "x": 0, "y": 0,
                "resize_source": False,
                "mask": ["16", 0],
            },
        },
        # JoinImageWithAlpha also expects 1 = transparent, so use LoadImage's mask as-is.
        "18": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["17", 0], "alpha": ["4", 1]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "test_full"}},
    }
    return wf


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "txt2img"
    if mode == "txt2img":
        wf = txt2img()
    elif mode == "crop_inpaint":
        wf = crop_inpaint()
    elif mode == "full":
        wf = full()
    else:
        wf = inpaint()
    res = post_prompt(wf)
    print("submitted", mode, res)
    pid = res["prompt_id"]
    hist = wait_for(pid)
    print(json.dumps(hist.get("outputs", {}), indent=2))
    print("status", hist.get("status", {}))
