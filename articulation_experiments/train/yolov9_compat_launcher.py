"""Launch official YOLOv9 scripts with current-PyTorch compatibility."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import torch
from PIL import ImageFont


def patch_pillow_font_getsize() -> None:
    """Restore the legacy Pillow API used by the official YOLOv9 plots."""
    if hasattr(ImageFont.FreeTypeFont, "getsize"):
        return

    def getsize(font, text, *args, **kwargs):
        left, top, right, bottom = font.getbbox(text, *args, **kwargs)
        return right - left, bottom - top

    ImageFont.FreeTypeFont.getsize = getsize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov9-root", type=Path, required=True)
    parser.add_argument("--script", default="train_dual.py")
    args, forwarded = parser.parse_known_args()
    root = args.yolov9_root.expanduser().resolve()
    script = (root / args.script).resolve()
    if not script.is_file() or root not in script.parents:
        raise FileNotFoundError(script)

    config_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "yolov9_config"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLOV5_CONFIG_DIR"] = str(config_dir)
    sys.path.insert(0, str(root))

    original_load = torch.load

    def compatible_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = compatible_load
    patch_pillow_font_getsize()
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
