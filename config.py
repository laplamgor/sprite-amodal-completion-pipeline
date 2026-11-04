"""Paths and connection settings for the pipeline scripts.

Everything is resolved from this file's location, so the scripts work from any
working directory and the whole pipeline folder can be moved or renamed without
editing them.

    sprite-amodal-completion-pipeline/config.py  ->  ROOT = <portable ComfyUI folder>
                        INPUT  = ./input      (sources, served by --input-directory)
                        OUTPUT = ./output     (renders + caches, --output-directory)

`input/` and `output/` live inside this repo so the data sits next to the code
and is gitignored. Sources stay flat in `input/`; every derived artefact
(axes, bust, depth, profiles) is written under `output/` so the caches sit with
the renders rather than mixed into the sources. The server has to be told about
both folders: `start_server.bat` passes `--input-directory` and
`--output-directory`, and `LoadImage`/`SaveImage` then resolve against these
paths. Without those flags ComfyUI uses its own `ComfyUI/input` and
`ComfyUI/output` and every render fails to find its source.

Override ROOT with the FLOW_ROOT environment variable if the scripts ever live
outside the portable install.
"""
import os
from pathlib import Path

_env_root = os.environ.get("FLOW_ROOT")
ROOT = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parent.parent

HERE = Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"

# --- inputs -------------------------------------------------------------
# Source PNGs live directly in input/, not in an incoming/ subfolder: the
# folder the server is pointed at is the folder you drop images into.
# `--dir` still takes a subfolder name for side batches.
INPUT = str(HERE / "input")                   # served by --input-directory

# --- outputs ------------------------------------------------------------
# Everything derived goes under output/ next to the renders, so there is one
# directory to clear or archive and the sources stay a flat folder of PNGs.
OUTPUT = str(HERE / "output")                    # <stem>_amodal.png
AXES = str(HERE / "output" / "axes")             # vlm_axes.py + rederive.py
BUST = str(HERE / "output" / "bust")             # bust_measure.py
PROFILES = str(HERE / "output" / "profiles")     # image_profile.py cache
DEPTH = str(HERE / "output" / "depth")           # depth_prepass.py
WORKFLOWS = str(COMFY / "user" / "default" / "workflows")  # generated UI graph

# --- weights ------------------------------------------------------------
# Kept in the repo next to the code, not in the ComfyUI model host: only these
# scripts load them.
MODELS = str(HERE / "models")                    # detector + tagger weights

# --- server -------------------------------------------------------------
HOST = "http://127.0.0.1:8199"

# --- custom nodes -------------------------------------------------------
# The pack lives in this repo, not in ComfyUI/custom_nodes; comfy_nodes.yaml
# tells ComfyUI where to find it (passed by start_server.bat).
CUSTOM_NODES = str(HERE / "comfy_nodes" / "amodal_prep")
NODES_YAML = str(HERE / "comfy_nodes.yaml")
