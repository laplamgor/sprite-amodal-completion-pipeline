"""Paths and connection settings for the pipeline scripts.

Everything is resolved from this file's location, so the scripts work from any
working directory and the whole pipeline folder can be moved or renamed without
editing them.

    sprite-amodal-completion-pipeline/config.py  ->  ROOT = <portable ComfyUI folder>
                        INPUT  = ../ComfyUI/input

Override ROOT with the FLOW_ROOT environment variable if the scripts ever live
outside the portable install.
"""
import os
from pathlib import Path

_env_root = os.environ.get("FLOW_ROOT")
ROOT = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parent.parent

COMFY = ROOT / "ComfyUI"

# --- inputs -------------------------------------------------------------
INPUT = str(COMFY / "input")                 # ../ComfyUI/input
INCOMING = str(COMFY / "input" / "incoming")  # default --dir
AXES = str(COMFY / "input" / "axes")          # vlm_axes.py output
BUST = str(COMFY / "input" / "bust")          # bust_measure.py output
PROFILES = str(COMFY / "input" / "profiles")  # image_profile.py cache
DEPTH = str(COMFY / "input" / "depth")        # depth_prepass.py output

# --- outputs ------------------------------------------------------------
OUTPUT = str(COMFY / "output")                          # <stem>_amodal.png
WORKFLOWS = str(COMFY / "user" / "default" / "workflows")  # generated UI graph

# --- server -------------------------------------------------------------
HOST = "http://127.0.0.1:8199"

# --- custom nodes -------------------------------------------------------
CUSTOM_NODES = str(COMFY / "custom_nodes" / "nude_prep")
