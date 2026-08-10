#!/usr/bin/env python3
"""Prepare grouped BPSD JPEG pages for the legacy OMR25 runtime.

The legacy runtime expects one directory per piece, a same-named JSON config,
and page images under ``imgs/<piece>_<page>/<piece>_<page>.png``.  BPSD page
files already contain both a piece prefix and a numeric page suffix, so this
script can build that structure without moving or deleting the source JPEGs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PAGE_NAME_RE = re.compile(r"^(?P<piece>.+)-(?P<page>\d+)$")


@dataclass(frozen=True)
class SourcePage:
    piece_name: str
    page_number: int
    source_path: Path


def discover_pages(source_dir: Path) -> dict[str, list[SourcePage]]:
    """Group direct child images by the final ``-<page number>`` suffix."""

    groups: dict[str, list[SourcePage]] = defaultdict(list)
    unmatched: list[str] = []
    seen: set[tuple[str, int]] = set()
    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = PAGE_NAME_RE.fullmatch(source_path.stem)
        if match is None:
            unmatched.append(source_path.name)
            continue
        piece_name = match.group("piece")
        page_number = int(match.group("page"))
        if page_number < 1:
            raise ValueError(f"Page number must be positive: {source_path.name}")
        key = (piece_name.casefold(), page_number)
        if key in seen:
            raise ValueError(
                f"Duplicate page {page_number} for piece {piece_name}: {source_path.name}"
            )
        seen.add(key)
        groups[piece_name].append(SourcePage(piece_name, page_number, source_path))

    if unmatched:
        joined = ", ".join(unmatched[:10])
        suffix = " ..." if len(unmatched) > 10 else ""
        raise ValueError(
            "Image names must end in '-<page number>'. Unmatched: "
            f"{joined}{suffix}"
        )
    if not groups:
        raise ValueError(f"No supported images found directly under {source_dir}")

    for piece_name, pages in groups.items():
        pages.sort(key=lambda page: page.page_number)
        actual = [page.page_number for page in pages]
        expected = list(range(1, len(pages) + 1))
        if actual != expected:
            raise ValueError(
                f"Pages for {piece_name} are not contiguous from 1: {actual}"
            )
    return dict(sorted(groups.items(), key=lambda item: item[0].casefold()))


def parse_time_signature(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError("Use a time signature such as 4/4 or 3/8")
    numerator, denominator = (int(match.group(1)), int(match.group(2)))
    if numerator < 1 or denominator not in {1, 2, 4, 8, 16, 32, 64}:
        raise argparse.ArgumentTypeError(f"Unsupported time signature: {value}")
    return numerator, denominator


def load_template(template_path: Path) -> dict:
    with template_path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    required = {"score_mode", "numTrack", "track_shift", "clef_options"}
    missing = sorted(required.difference(template))
    if missing:
        raise ValueError(f"Piano config template is missing: {', '.join(missing)}")
    if template["score_mode"] != "piano" or int(template["numTrack"]) != 2:
        raise ValueError("Template must use score_mode='piano' and numTrack=2")
    return template


def build_config(
    template: dict,
    page_count: int,
    time_signature: tuple[int, int],
    source_piece_name: str,
) -> dict:
    # JSON round-trip gives us a simple deep copy without another dependency.
    config = json.loads(json.dumps(template))
    config["numPage"] = page_count
    config["numPerPage"] = 1
    config["rotate"] = False
    config["score_mode"] = "piano"
    config["numTrack"] = 2
    config["track_shift"] = [0, 0]
    config["clef_options"] = [[1], [-1]]
    config["tsChange"] = [
        {
            "page": 1,
            "loc": [0, 0],
            "time_signature": list(time_signature),
            "mov": 1,
        }
    ]
    config["_bpsd_import"] = {
        "source_piece_name": source_piece_name,
        "time_signature_status": "needs_manual_review",
        "note": "Verify tsChange before MusicXML conversion.",
    }
    return config


def write_json(path: Path, document: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def convert_page(source_path: Path, destination_path: Path, overwrite: bool) -> str:
    if destination_path.exists() and not overwrite:
        with Image.open(source_path) as source_image, Image.open(destination_path) as existing:
            if source_image.size != existing.size:
                raise FileExistsError(
                    f"Existing page has a different size: {destination_path}"
                )
        return "existing"

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f"{destination_path.stem}.tmp.png")
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"1", "L", "RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(temporary, format="PNG", compress_level=1)
    temporary.replace(destination_path)
    return "written"


def import_groups(
    groups: dict[str, list[SourcePage]],
    output_root: Path,
    template: dict,
    time_signature: tuple[int, int],
    apply: bool,
    overwrite: bool,
) -> dict:
    manifest_groups = []
    written = 0
    existing = 0
    for piece_name, pages in groups.items():
        target_root = output_root / piece_name
        page_records = []
        for output_number, page in enumerate(pages, start=1):
            image_id = f"{piece_name}_{output_number}"
            destination = target_root / "imgs" / image_id / f"{image_id}.png"
            if apply:
                status = convert_page(page.source_path, destination, overwrite)
                written += status == "written"
                existing += status == "existing"
            page_records.append(
                {
                    "output_page": output_number,
                    "source_page": page.page_number,
                    "source_file": page.source_path.name,
                    "destination": str(destination.resolve()),
                }
            )

        config_path = target_root / f"{piece_name}.json"
        if apply:
            target_root.mkdir(parents=True, exist_ok=True)
            config = build_config(template, len(pages), time_signature, piece_name)
            if config_path.exists() and not overwrite:
                existing_config = json.loads(config_path.read_text(encoding="utf-8"))
                if existing_config.get("numPage") != len(pages):
                    raise FileExistsError(
                        f"Existing config has a different numPage: {config_path}"
                    )
            else:
                write_json(config_path, config)
        manifest_groups.append(
            {
                "piece_name": piece_name,
                "page_count": len(pages),
                "config": str(config_path.resolve()),
                "pages": page_records,
            }
        )

    manifest = {
        "schema_version": 1,
        "applied": apply,
        "piece_count": len(groups),
        "page_count": sum(len(pages) for pages in groups.values()),
        "default_time_signature": list(time_signature),
        "time_signature_status": "needs_manual_review",
        "written_page_count": written,
        "existing_page_count": existing,
        "pieces": manifest_groups,
    }
    if apply:
        write_json(output_root / "bpsd_import_manifest.json", manifest)
        write_json(
            output_root / "piecesToRun.bpsd.json",
            [item["piece_name"] for item in manifest_groups],
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Group BPSD page images and prepare OMR25 piano input folders."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "string_dataset" / "pdf_data" / "BPSD_score_scan_jpeg",
        help="Directory containing the original BPSD page images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "string_dataset" / "pdf_data",
        help="OMR25 pdf_data directory that will receive one folder per piece.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=repo_root / "pianoConfigTemplate.json",
        help="Piano JSON template.",
    )
    parser.add_argument(
        "--time-signature",
        type=parse_time_signature,
        default=parse_time_signature("4/4"),
        help="Temporary initial time signature for generated configs (default: 4/4).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create files. Without this flag the command only prints a dry-run plan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously imported PNG/config files. Source images are never changed.",
    )
    return parser


def print_summary(manifest: dict, source_dir: Path, output_root: Path) -> None:
    action = "IMPORT" if manifest["applied"] else "DRY RUN"
    print(f"{action}: {manifest['piece_count']} pieces, {manifest['page_count']} pages")
    print(f"Source: {source_dir.resolve()}")
    print(f"Output: {output_root.resolve()}")
    for item in manifest["pieces"]:
        print(f"  {item['piece_name']}: {item['page_count']} pages")
    print("All generated configs require manual tsChange/time-signature review.")
    if not manifest["applied"]:
        print("No files were created. Re-run with --apply after reviewing this plan.")
    else:
        print(
            "Pages written: "
            f"{manifest['written_page_count']}; existing reused: "
            f"{manifest['existing_page_count']}"
        )
        print(f"Manifest: {(output_root / 'bpsd_import_manifest.json').resolve()}")
        print(f"Piece list: {(output_root / 'piecesToRun.bpsd.json').resolve()}")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Template does not exist: {template_path}")
    if args.apply:
        output_root.mkdir(parents=True, exist_ok=True)
    groups = discover_pages(source_dir)
    template = load_template(template_path)
    manifest = import_groups(
        groups,
        output_root,
        template,
        args.time_signature,
        apply=args.apply,
        overwrite=args.overwrite,
    )
    print_summary(manifest, source_dir, output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
