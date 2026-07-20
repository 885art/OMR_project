"""Geometry and image helpers for deterministic overlapping YOLO tiles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"Expected four bbox values, got {len(values)}")
        bbox = cls(*(float(value) for value in values))
        if not all(math.isfinite(value) for value in bbox.as_list()):
            raise ValueError(f"BBox contains a non-finite value: {values}")
        return bbox

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def is_positive(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    size: int

    @property
    def x2(self) -> int:
        return self.x + self.size

    @property
    def y2(self) -> int:
        return self.y + self.size

    @property
    def bbox(self) -> BBox:
        return BBox(float(self.x), float(self.y), float(self.x2), float(self.y2))


@dataclass(frozen=True)
class Assignment:
    clipped_source_bbox: BBox
    tile_bbox: BBox
    intersection_ratio: float
    center_inside: bool
    clipped: bool

    @property
    def reason(self) -> str:
        if self.center_inside and self.intersection_ratio >= 1.0 - 1e-12:
            return "center_inside_full_bbox"
        if self.center_inside:
            return "center_inside_clipped_bbox"
        return "intersection_ratio"


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return fixed-stride starts; the final tile is padded instead of shifted."""
    if length <= 0:
        raise ValueError(f"Image dimension must be positive, got {length}")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride cannot exceed tile_size")
    return list(range(0, length, stride))


def generate_tile_windows(
    source_width: int,
    source_height: int,
    tile_size: int,
    stride: int,
) -> list[TileWindow]:
    return [
        TileWindow(x=x, y=y, size=tile_size)
        for y in tile_starts(source_height, tile_size, stride)
        for x in tile_starts(source_width, tile_size, stride)
    ]


def intersection(a: BBox, b: BBox) -> BBox | None:
    clipped = BBox(
        max(a.x1, b.x1),
        max(a.y1, b.y1),
        min(a.x2, b.x2),
        min(a.y2, b.y2),
    )
    return clipped if clipped.is_positive else None


def assign_bbox(
    bbox: BBox,
    tile: TileWindow,
    minimum_intersection_ratio: float,
) -> Assignment | None:
    """Assign by center inclusion or by retained bbox-area ratio."""
    if not bbox.is_positive:
        return None
    if not 0.0 <= minimum_intersection_ratio <= 1.0:
        raise ValueError("minimum_intersection_ratio must be in [0, 1]")

    tile_region = tile.bbox
    clipped = intersection(bbox, tile_region)
    if clipped is None:
        return None

    center_x, center_y = bbox.center
    center_inside = (
        tile_region.x1 <= center_x < tile_region.x2
        and tile_region.y1 <= center_y < tile_region.y2
    )
    ratio = clipped.area / bbox.area
    if not center_inside and ratio < minimum_intersection_ratio:
        return None

    tile_bbox = BBox(
        clipped.x1 - tile.x,
        clipped.y1 - tile.y,
        clipped.x2 - tile.x,
        clipped.y2 - tile.y,
    )
    return Assignment(
        clipped_source_bbox=clipped,
        tile_bbox=tile_bbox,
        intersection_ratio=ratio,
        center_inside=center_inside,
        clipped=any(
            abs(source - retained) > 1e-9
            for source, retained in zip(bbox.as_list(), clipped.as_list())
        ),
    )


def yolo_box(bbox: BBox, tile_size: int) -> tuple[float, float, float, float]:
    if not bbox.is_positive:
        raise ValueError(f"Cannot normalize non-positive bbox: {bbox}")
    center_x, center_y = bbox.center
    values = (
        center_x / tile_size,
        center_y / tile_size,
        bbox.width / tile_size,
        bbox.height / tile_size,
    )
    epsilon = 1e-9
    if any(value < -epsilon or value > 1.0 + epsilon for value in values):
        raise ValueError(f"Normalized bbox falls outside tile: {values}")
    return tuple(min(1.0, max(0.0, value)) for value in values)


def crop_and_pad(
    source: Image.Image,
    tile: TileWindow,
    source_width: int,
    source_height: int,
) -> Image.Image:
    """Crop a tile and pad its right/bottom edges with white pixels."""
    right = min(tile.x2, source_width)
    bottom = min(tile.y2, source_height)
    cropped = source.crop((tile.x, tile.y, right, bottom))
    if source.mode == "RGB":
        fill = (255, 255, 255)
    elif source.mode == "RGBA":
        fill = (255, 255, 255, 255)
    else:
        fill = 255
    padded = Image.new(source.mode, (tile.size, tile.size), fill)
    padded.paste(cropped, (0, 0))
    return padded


def safe_tile_stem(source_filename: str, image_id: str | int, tile: TileWindow) -> str:
    source_stem = Path(source_filename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("_")
    return f"{safe_stem}__id{image_id}__x{tile.x:04d}_y{tile.y:04d}"
