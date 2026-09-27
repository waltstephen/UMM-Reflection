#!/usr/bin/env python3
"""Write ``parquet_info.json`` for a directory of parquet shards.

BAGEL's parquet readers key row-group metadata by the absolute shard path, so
the file must be regenerated whenever a dataset directory is moved to a new
machine or mount point. Existing entries are replaced.

Usage:
  python scripts/data/write_parquet_info.py data/sft/trajectory_parquet
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        root = os.path.abspath(str(directory))
        info = {}
        for name in sorted(os.listdir(root)):
            if not name.endswith(".parquet"):
                continue
            path = os.path.join(root, name)
            metadata = pq.ParquetFile(path).metadata
            info[path] = {
                "num_row_groups": metadata.num_row_groups,
                "num_rows": metadata.num_rows,
            }
        if not info:
            raise SystemExit(f"no parquet shards under {root}")
        target = Path(root) / "parquet_info.json"
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        rows = sum(item["num_rows"] for item in info.values())
        print(f"{target}: {len(info)} shards, {rows} rows")


if __name__ == "__main__":
    main()
