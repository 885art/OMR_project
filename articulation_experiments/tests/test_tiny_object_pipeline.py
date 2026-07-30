from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from articulation_experiments.dataset.merge_deepscores_complete import merge_split
from articulation_experiments.inference.infer_tiled_page import tiled_predict


class FakeYoloV9:
    names = {0: "articStaccatoAbove"}

    def __init__(self) -> None:
        self.received_sizes: list[tuple[int, int]] = []

    def predict_tiles(self, tiles, *, confidence):
        self.received_sizes.extend(tile.size for tile in tiles)
        return [
            [
                {
                    "bbox_xyxy": [100.0, 120.0, 200.0, 220.0],
                    "class_id": 0,
                    "confidence": 0.9,
                }
            ]
            for _ in tiles
        ]


def test_native_yolov9_resizes_input_and_restores_source_coordinates() -> None:
    model = FakeYoloV9()
    page = Image.new("RGB", (400, 300), "white")

    raw, _ = tiled_predict(
        model,
        page,
        "page",
        "page.png",
        tile_size=512,
        overlap=128,
        confidence=0.1,
        device="cpu",
        batch=2,
        model_input_size=1024,
        edge_policy="shift",
    )

    assert model.received_sizes == [(1024, 1024)]
    assert raw["model_input_size"] == 1024
    assert raw["model_input_scale"] == 2.0
    assert raw["predictions"][0]["bbox_xyxy"] == [50.0, 60.0, 100.0, 110.0]


def write_shard(
    path: Path,
    category_id: int,
    image_id: int,
    filename: str,
    annotation_id: int,
) -> None:
    categories = {
        "73": {
            "name": "articStaccatoAbove",
            "annotation_set": "deepscores",
        },
        "74": {
            "name": "articStaccatoBelow",
            "annotation_set": "deepscores",
        },
    }
    document = {
        "info": {},
        "annotation_sets": ["deepscores"],
        "categories": categories,
        "images": [
            {
                "id": image_id,
                "filename": filename,
                "width": 100,
                "height": 120,
                "ann_ids": [str(annotation_id)],
            }
        ],
        "annotations": {
            str(annotation_id): {
                "a_bbox": [10, 20, 16, 26],
                "cat_id": [str(category_id)],
                "img_id": str(image_id),
            }
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_complete_shards_are_deduplicated_by_page(tmp_path: Path) -> None:
    classes = [
        {
            "deepscores_id": 73,
            "deepscores_name": "articStaccatoAbove",
            "yolo_id": 0,
        },
        {
            "deepscores_id": 74,
            "deepscores_name": "articStaccatoBelow",
            "yolo_id": 1,
        },
    ]
    for class_id in (73, 74):
        write_shard(
            tmp_path / f"deepscores-complete-{class_id}_train.json",
            category_id=73,
            image_id=1,
            filename="same-page.png",
            annotation_id=100,
        )

    document, statistics = merge_split(
        tmp_path,
        "train",
        classes,
        {"73", "74"},
        max_shards=None,
        max_images_per_shard=None,
    )

    assert len(document["images"]) == 1
    assert len(document["annotations"]) == 1
    assert statistics["unique_page_count"] == 1
    assert statistics["counters"]["duplicate_page_occurrences"] == 1
