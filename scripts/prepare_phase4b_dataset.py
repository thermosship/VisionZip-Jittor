#!/usr/bin/env python3
"""Prepare a pinned, attributed CommonCatalog CC-BY subset for Phase 4B.

The default invocation is a no-download preflight. Pass ``--execute`` only after
reviewing the source size, disk estimate, revision, and output location.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionzip_jittor.phase4b_config import load_phase4b_config
from visionzip_jittor.phase4b_data import (
    assign_exact_splits,
    file_sha256,
    prepared_sample_from_row,
    preflight_report,
    source_image_bytes,
    source_row_rejection,
    source_sample_id,
    write_prepared_dataset_manifest,
)


REQUIRED_COLUMNS = (
    "jpg",
    "blip2_caption",
    "status",
    "licensename",
    "licenseurl",
    "width",
    "height",
    "photoid",
    "uid",
    "unickname",
    "title",
    "pageurl",
    "sha256",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets/phase4b/commoncatalog_cc_by_8k",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/cache/huggingface"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional directory containing the pinned Parquet paths.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download/read shards and materialize the prepared dataset.",
    )
    return parser.parse_args()


def resolve_source_path(shard, config, cache_dir: Path, source_dir: Path):
    if source_dir is not None:
        path = source_dir / shard.path
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source shard is missing: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for --execute without --source-dir"
        ) from error
    return Path(
        hf_hub_download(
            repo_id=config.dataset.dataset_id,
            repo_type="dataset",
            filename=shard.path,
            revision=config.dataset.revision,
            cache_dir=str(cache_dir),
        )
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(str(temporary), str(path))


def validate_image(content: bytes, min_dimension: int):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to prepare Phase 4B images") from error
    from io import BytesIO

    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        width, height = image.size
        image_format = str(image.format or "").upper()
    if min(width, height) < min_dimension:
        raise ValueError("decoded image is smaller than min_dimension")
    if image_format not in {"JPEG", "JPG"}:
        raise ValueError(f"expected JPEG bytes in the jpg field, got {image_format}")
    return int(width), int(height)


def print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = parse_args()
    config = load_phase4b_config(args.config)
    disk_parent = args.output_dir.parent
    disk_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(disk_parent).free
    preflight = preflight_report(config, free_bytes=free_bytes)
    preflight.update(
        {
            "config": str(args.config.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "execute_requested": bool(args.execute),
        }
    )
    print("=" * 72)
    print("Phase 4B dataset preflight")
    print("=" * 72)
    print_json(preflight)
    if not preflight["disk_preflight_passed"]:
        raise SystemExit("Phase 4B disk preflight failed")
    if not args.execute:
        print("Preflight only: no network download or dataset materialization was performed.")
        return
    if (args.output_dir / "manifest.json").exists():
        raise SystemExit(
            "Prepared manifest already exists; use a new output directory or remove the generated directory explicitly."
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for Phase 4B preparation") from error

    samples = []
    seen_sample_ids = set()
    seen_source_hashes = set()
    rejection_counts = Counter()
    source_files = []
    target = config.dataset.target_sample_count

    for shard in config.dataset.source_shards:
        source_path = resolve_source_path(
            shard,
            config,
            args.cache_dir,
            args.source_dir,
        )
        actual_size = source_path.stat().st_size
        if actual_size != shard.size_bytes:
            raise ValueError(
                f"Pinned source size mismatch for {shard.path}: "
                f"expected {shard.size_bytes}, got {actual_size}"
            )
        source_hash = file_sha256(source_path)
        parquet = pq.ParquetFile(source_path)
        if parquet.metadata.num_rows != shard.num_rows:
            raise ValueError(
                f"Pinned source row-count mismatch for {shard.path}: "
                f"expected {shard.num_rows}, got {parquet.metadata.num_rows}"
            )
        available = set(parquet.schema_arrow.names)
        missing = sorted(set(REQUIRED_COLUMNS) - available)
        if missing:
            raise ValueError(f"Pinned source shard lacks required columns: {missing}")
        source_files.append(
            {
                **shard.to_dict(),
                "local_path": str(source_path),
                "sha256": source_hash,
            }
        )
        source_row_offset = 0
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group, columns=list(REQUIRED_COLUMNS))
            for local_row, row in enumerate(table.to_pylist()):
                source_row = source_row_offset + local_row
                reason = source_row_rejection(row, config)
                if reason is not None:
                    rejection_counts[reason] += 1
                    continue
                sample_id = source_sample_id(row)
                source_hash_value = str(row.get("sha256") or "").lower()
                if sample_id in seen_sample_ids or source_hash_value in seen_source_hashes:
                    rejection_counts["duplicate"] += 1
                    continue
                try:
                    image_bytes = source_image_bytes(row[config.dataset.image_field])
                    decoded_width, decoded_height = validate_image(
                        image_bytes,
                        config.dataset.min_dimension,
                    )
                except Exception:
                    rejection_counts["image_decode"] += 1
                    continue
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if source_hash_value and image_hash != source_hash_value:
                    rejection_counts["source_sha256"] += 1
                    continue
                if decoded_width != int(row["width"]) or decoded_height != int(row["height"]):
                    rejection_counts["dimension_metadata"] += 1
                    continue
                relative_image = Path("images") / sample_id[:8] / f"{sample_id}.jpg"
                image_path = args.output_dir / relative_image
                if image_path.exists():
                    if file_sha256(image_path) != image_hash:
                        raise ValueError(f"Existing prepared image hash mismatch: {image_path}")
                else:
                    atomic_write_bytes(image_path, image_bytes)
                sample = prepared_sample_from_row(
                    row=row,
                    config=config,
                    source_shard=shard.path,
                    source_row=source_row,
                    image_path=relative_image.as_posix(),
                    image_sha256=image_hash,
                )
                sample.validate()
                samples.append(sample)
                seen_sample_ids.add(sample.sample_id)
                seen_source_hashes.add(sample.source_image_sha256)
                if len(samples) >= target:
                    break
            source_row_offset += table.num_rows
            if len(samples) >= target:
                break
        if len(samples) >= target:
            break

    if len(samples) != target:
        raise RuntimeError(
            f"Only {len(samples)} acceptable rows were found; target is {target}. "
            f"Rejections: {dict(rejection_counts)}"
        )
    split_samples = assign_exact_splits(
        samples,
        config.dataset.validation_sample_count,
        config.dataset.split_seed,
    )
    manifest = write_prepared_dataset_manifest(
        args.output_dir,
        config,
        split_samples,
        source_files,
        rejection_counts,
    )
    result = {
        "artifact_type": "phase4b_dataset_preparation_result_v1",
        "passed": True,
        "preflight": preflight,
        "manifest": manifest.to_dict(),
        "manifest_path": str((args.output_dir / "manifest.json").resolve()),
    }
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("=" * 72)
    print("Phase 4B dataset preparation complete")
    print("=" * 72)
    print_json(result)


if __name__ == "__main__":
    main()
