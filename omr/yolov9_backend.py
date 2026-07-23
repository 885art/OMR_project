"""Adapter for the official WongKinYiu/yolov9 PyTorch implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def default_yolov9_root() -> Path:
    return Path(__file__).resolve().parents[2] / "yolov9"


def _patch_torch_load() -> None:
    """Keep pre-PyTorch-2.6 YOLOv9 checkpoints loadable on current PyTorch."""

    if getattr(torch.load, "_omr_yolov9_compatible", False):
        return
    original = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    compatible_load._omr_yolov9_compatible = True  # type: ignore[attr-defined]
    torch.load = compatible_load  # type: ignore[assignment]


def _prepare_imports(yolov9_root: Path) -> None:
    root = yolov9_root.expanduser().resolve()
    if not (root / "models" / "common.py").is_file():
        raise FileNotFoundError(
            f"Official YOLOv9 repository not found: {root}. "
            "Clone https://github.com/WongKinYiu/yolov9 beside 25-omr."
        )
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "articulation_experiments"
        / "outputs"
        / "yolov9_config"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLOV5_CONFIG_DIR", str(config_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _patch_torch_load()


class YoloV9Detector:
    """Expose official YOLOv9 detections in the interface used by tiled OMR."""

    backend_name = "yolov9"

    def __init__(
        self,
        weights: str | Path,
        data_yaml: str | Path,
        *,
        yolov9_root: str | Path | None = None,
        device: str | int | None = None,
        half: bool = True,
    ) -> None:
        root = (
            Path(yolov9_root).expanduser().resolve()
            if yolov9_root is not None
            else default_yolov9_root()
        )
        _prepare_imports(root)
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        requested_device = "" if device is None else str(device)
        selected_device = select_device(requested_device)
        use_half = bool(half and selected_device.type != "cpu")
        self.model = DetectMultiBackend(
            str(Path(weights).expanduser().resolve()),
            device=selected_device,
            data=str(Path(data_yaml).expanduser().resolve()),
            fp16=use_half,
        )
        self.device = self.model.device
        self.fp16 = bool(self.model.fp16)
        self.names = {
            int(key): str(value)
            for key, value in (
                self.model.names.items()
                if isinstance(self.model.names, dict)
                else enumerate(self.model.names)
            )
        }
        self._nms = __import__(
            "utils.general", fromlist=["non_max_suppression"]
        ).non_max_suppression
        self._warmed_shapes: set[tuple[int, int, int, int]] = set()

    def predict_tiles(
        self,
        tiles: list[Image.Image],
        *,
        confidence: float,
        iou: float = 0.45,
        max_det: int = 2000,
    ) -> list[list[dict[str, Any]]]:
        if not tiles:
            return []
        arrays = [
            np.asarray(tile.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
            for tile in tiles
        ]
        batch = np.ascontiguousarray(np.stack(arrays))
        tensor = torch.from_numpy(batch).to(self.device)
        tensor = tensor.half() if self.fp16 else tensor.float()
        tensor /= 255.0
        shape = tuple(int(value) for value in tensor.shape)
        if shape not in self._warmed_shapes:
            self.model.warmup(imgsz=shape)
            self._warmed_shapes.add(shape)
        with torch.inference_mode():
            raw = self.model(tensor)
            # Official YOLOv9 dual-head checkpoints return
            # ([auxiliary, primary], training_outputs).
            if (
                isinstance(raw, (list, tuple))
                and raw
                and isinstance(raw[0], (list, tuple))
            ):
                prediction = raw[0][1]
            elif isinstance(raw, (list, tuple)):
                prediction = raw[0]
            else:
                prediction = raw
            detections = self._nms(
                prediction,
                float(confidence),
                float(iou),
                max_det=int(max_det),
            )
        results: list[list[dict[str, Any]]] = []
        for detected in detections:
            one_tile = []
            for x0, y0, x1, y1, conf, class_id in detected.detach().float().cpu().tolist():
                one_tile.append(
                    {
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "confidence": conf,
                        "class_id": int(class_id),
                    }
                )
            results.append(one_tile)
        return results


def load_yolov9_detector(
    weights: str | Path,
    data_yaml: str | Path,
    *,
    yolov9_root: str | Path | None = None,
    device: str | int | None = None,
) -> YoloV9Detector:
    return YoloV9Detector(
        weights,
        data_yaml,
        yolov9_root=yolov9_root,
        device=device,
    )
