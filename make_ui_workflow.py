"""Generate the ComfyUI UI workflow using the `nude_prep` custom nodes.

With SafeFrame / MaskUnionFill / Finalize the graph now matches the script:
per-image safe frame + aspect clamp, per-component hole filling, and an exact
resize back to source dimensions.

    python make_ui_workflow.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C      # noqa: E402
import run_folder as rf  # single source of truth for prompts and model files

HOST = C.HOST
OUT = os.path.join(C.WORKFLOWS, "qwen_amodal_batch.json")

WIDGET_TYPES = {"COMBO", "INT", "FLOAT", "STRING", "BOOLEAN"}

OI = None          # node schemas, fetched by main(); import stays side-effect free


def fetch_schemas():
    global OI
    with urllib.request.urlopen(HOST + "/object_info", timeout=60) as r:
        OI = json.loads(r.read())
    return OI


def schema(t):
    d = OI[t]
    req = d["input"].get("required", {})
    opt = d["input"].get("optional", {})
    return list(req.items()) + list(opt.items()), d


def is_socket(spec):
    t = spec[0]
    return not (isinstance(t, list) or t in WIDGET_TYPES)


class WF:
    def __init__(self):
        self.nodes, self.links = [], []
        self.nid = self.lid = 1

    def add(self, ntype, pos, values=None, size=(320, 130)):
        order, d = schema(ntype)
        values = values or {}
        n = {"id": self.nid, "type": ntype, "pos": list(pos), "size": list(size),
             "flags": {}, "order": len(self.nodes), "mode": 0,
             "inputs": [], "outputs": [],
             "properties": {"Node name for S&R": ntype}, "widgets_values": []}

        for name, spec in order:
            if is_socket(spec):
                n["inputs"].append({"name": name, "type": spec[0], "link": None})
            else:
                val = values.get(name, spec[1].get("default") if len(spec) > 1 else None)
                if val is None and isinstance(spec[0], list):
                    val = spec[0][0]
                n["widgets_values"].append(val)

        if ntype == "KSampler":
            n["widgets_values"].insert(1, "fixed")
        if ntype == "LoadImage":
            n["widgets_values"].append("image")

        for i, otype in enumerate(d.get("output") or []):
            names = d.get("output_name") or []
            n["outputs"].append({"name": names[i] if i < len(names) else otype,
                                 "type": otype, "links": [], "slot_index": i})
        self.nodes.append(n)
        self.nid += 1
        return n

    def connect(self, src, src_slot, dst, dst_name):
        si = next(i for i, x in enumerate(dst["inputs"]) if x["name"] == dst_name)
        lid = self.lid
        self.links.append([lid, src["id"], src_slot, dst["id"], si,
                           src["outputs"][src_slot]["type"]])
        dst["inputs"][si]["link"] = lid
        src["outputs"][src_slot]["links"].append(lid)
        self.lid += 1

    def dump(self):
        return {"last_node_id": self.nid - 1, "last_link_id": self.lid - 1,
                "nodes": self.nodes, "links": self.links, "groups": [],
                "config": {}, "extra": {"ds": {"scale": 0.5, "offset": [40, 40]}},
                "version": 0.4}


PROMPT = rf.PROMPT


def build():
    w = WF()

    unet = w.add("UnetLoaderGGUF", (-40, 40), {"unet_name": rf.UNET})
    clip = w.add("CLIPLoader", (-40, 190), {"clip_name": rf.TE,
                                            "type": "qwen_image", "device": "default"})
    vae = w.add("VAELoader", (-40, 350), {"vae_name": rf.VAE})
    patch = w.add("ModelPatchLoader", (-40, 490),
                  {"name": rf.CN})
    sam = w.add("CheckpointLoaderSimple", (-40, 630), {"ckpt_name": rf.SAM})
    biref = w.add("LoadBackgroundRemovalModel", (-40, 790), {"bg_removal_name": rf.BIREF})

    for bi in range(2):
        x = 380 + bi * 2100

        img = w.add("LoadImage", (x, 40), {"image": "sample4.png"}, size=(320, 400))
        frame = w.add("NudeSafeFrame", (x, 480),
                      {"margin": 0.08, "scale": 1.25, "max_megapixels": 1.5, "min_aspect": 0.62, "max_aspect": 1.60},
                      size=(300, 220))
        w.connect(img, 0, frame, "image")
        w.connect(img, 1, frame, "mask")     # LoadImage.mask = 1 - alpha

        p1 = w.add("CLIPTextEncode", (x, 760), {"text": "outfit"}, size=(320, 160))
        p2 = w.add("CLIPTextEncode", (x, 950), {"text": rf.GARMENTS}, size=(320, 160))
        w.connect(sam, 1, p1, "clip")
        w.connect(sam, 1, p2, "clip")

        d1 = w.add("SAM3_Detect", (x + 380, 720),
                   {"threshold": 0.30, "refine_iterations": 2, "individual_masks": False}, size=(320, 200))
        d2 = w.add("SAM3_Detect", (x + 380, 950),
                   {"threshold": 0.30, "refine_iterations": 2, "individual_masks": False}, size=(320, 200))
        w.connect(sam, 0, d1, "model")
        w.connect(frame, 0, d1, "image")
        w.connect(p1, 0, d1, "conditioning")
        w.connect(sam, 0, d2, "model")
        w.connect(frame, 0, d2, "image")
        w.connect(p2, 0, d2, "conditioning")

        fill = w.add("NudeMaskUnionFill", (x + 760, 720), {"grow": 16, "min_area": 400}, size=(300, 180))
        w.connect(d1, 0, fill, "mask1")
        w.connect(d2, 0, fill, "mask2")

        # protected features: face / hair / hands are subtracted from the mask
        e1 = w.add("CLIPTextEncode", (x + 380, 1080), {"text": "face, mouth, lips"}, size=(320, 120))
        e2 = w.add("CLIPTextEncode", (x + 380, 1220), {"text": "hair, braid"}, size=(320, 120))
        e3 = w.add("CLIPTextEncode", (x + 380, 1360), {"text": "hand, fingers"}, size=(320, 120))
        e4 = w.add("CLIPTextEncode", (x + 380, 1500), {"text": "shoe, boot, footwear, sandal"}, size=(320, 120))
        w.connect(sam, 1, e1, "clip")
        w.connect(sam, 1, e2, "clip")
        w.connect(sam, 1, e3, "clip")
        w.connect(sam, 1, e4, "clip")

        q1 = w.add("SAM3_Detect", (x + 760, 1060), {"threshold": rf.PROTECT_THRESHOLD, "refine_iterations": 2, "individual_masks": False}, size=(320, 180))
        q2 = w.add("SAM3_Detect", (x + 760, 1250), {"threshold": rf.PROTECT_THRESHOLD, "refine_iterations": 2, "individual_masks": False}, size=(320, 180))
        q3 = w.add("SAM3_Detect", (x + 760, 1440), {"threshold": rf.PROTECT_THRESHOLD, "refine_iterations": 2, "individual_masks": False}, size=(320, 180))
        q4 = w.add("SAM3_Detect", (x + 760, 1620), {"threshold": rf.PROTECT_THRESHOLD, "refine_iterations": 2, "individual_masks": False}, size=(320, 180))
        for det, enc in ((q1, e1), (q2, e2), (q3, e3), (q4, e4)):
            w.connect(sam, 0, det, "model")
            w.connect(frame, 0, det, "image")
            w.connect(enc, 0, det, "conditioning")

        prot = w.add("NudeMaskUnionFill", (x + 1120, 1250), {"grow": 4, "min_area": 100}, size=(300, 180))
        prot_sources = [q1]
        if rf.HAIR_PROTECT:
            prot_sources.append(q2)
        prot_sources.append(q3)
        if rf.FOOTWEAR_PROTECT:
            prot_sources.append(q4)
        for i, src in enumerate(prot_sources):
            w.connect(src, 0, prot, "mask%d" % (i + 1))

        # segformer parsing: catches garments SAM3 misses, but over-segments stylised
        # costumes, so it is opt-in (rf.PARSE)
        if rf.PARSE:
            parse_n = w.add("NudeParseClothes", (x + 760, 1620), {"grow": 6}, size=(300, 130))
            w.connect(frame, 0, parse_n, "image")

            add = w.add("MaskComposite", (x + 1460, 1400), {"x": 0, "y": 0, "operation": "add"},
                        size=(300, 150))
            w.connect(fill, 0, add, "destination")
            w.connect(parse_n, 0, add, "source")
            base = add
        else:
            base = fill

        sub = w.add("MaskComposite", (x + 1460, 1250), {"x": 0, "y": 0, "operation": "subtract"}, size=(300, 150))
        w.connect(base, 0, sub, "destination")
        w.connect(prot, 0, sub, "source")

        cn = w.add("QwenImageDiffsynthControlnet", (x + 1120, 520),
                   {"strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}, size=(320, 200))
        w.connect(unet, 0, cn, "model")
        w.connect(patch, 0, cn, "model_patch")
        w.connect(vae, 0, cn, "vae")
        w.connect(frame, 0, cn, "image")
        w.connect(sub, 0, cn, "mask")

        ve = w.add("VAEEncode", (x + 760, 1300))
        w.connect(frame, 0, ve, "pixels")
        w.connect(vae, 0, ve, "vae")

        # reference is resized to the sampling latent's exact size (stock node forces ~1 MP)
        enc = w.add("NudeTextEncode", (x + 760, 960), {"prompt": PROMPT}, size=(420, 340))
        w.connect(clip, 0, enc, "clip")
        w.connect(vae, 0, enc, "vae")
        w.connect(ve, 0, enc, "latent")
        w.connect(frame, 0, enc, "image1")

        nz = w.add("SetLatentNoiseMask", (x + 1120, 1300))
        w.connect(ve, 0, nz, "samples")
        w.connect(sub, 0, nz, "mask")

        ks = w.add("KSampler", (x + 1500, 560),
                   {"seed": 123456789, "steps": rf.STEPS, "cfg": 1.0,
                    "sampler_name": rf.SAMPLER, "scheduler": rf.SCHEDULER, "denoise": 1.0},
                   size=(320, 260))
        w.connect(cn, 0, ks, "model")
        w.connect(enc, 0, ks, "positive")
        w.connect(enc, 0, ks, "negative")
        w.connect(nz, 0, ks, "latent_image")

        dec = w.add("VAEDecode", (x + 1500, 880))
        w.connect(ks, 0, dec, "samples")
        w.connect(vae, 0, dec, "vae")

        rb = w.add("RemoveBackground", (x + 1500, 1020))
        w.connect(biref, 0, rb, "bg_removal_model")
        w.connect(dec, 0, rb, "image")

        fin = w.add("NudeFinalize", (x + 1500, 1180),
                      {"edge_soften": 0.6, "soft_alpha_below": 0.98, "bg_tolerance": 18.0}, size=(300, 230))
        w.connect(dec, 0, fin, "image")
        w.connect(rb, 0, fin, "mask")
        w.connect(frame, 1, fin, "frame")
        w.connect(img, 0, fin, "source_image")
        w.connect(img, 1, fin, "source_mask")
        w.connect(sub, 0, fin, "edited_mask")

        join = w.add("JoinImageWithAlpha", (x + 1500, 1400))
        w.connect(fin, 0, join, "image")
        w.connect(fin, 1, join, "alpha")   # Finalize already returns 1 = transparent

        save = w.add("SaveImage", (x + 1500, 1660), {"filename_prefix": "amodal_%d" % (bi + 1)},
                     size=(360, 420))
        w.connect(join, 0, save, "images")

    return w.dump()


def write():
    """Regenerate the UI workflow from the current script constants."""
    if OI is None:
        fetch_schemas()
    wf = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1)
    return OUT, len(wf["nodes"]), len(wf["links"])


if __name__ == "__main__":
    out, n, l = write()
    print("wrote", out, "| nodes", n, "links", l)
