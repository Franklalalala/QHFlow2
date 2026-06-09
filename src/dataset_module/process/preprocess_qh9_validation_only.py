from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def split_file_name(dataset_name: str, split: str) -> str:
    if dataset_name != "QH9Stable":
        raise ValueError("Only QH9Stable validation-only preprocessing is supported")
    if split == "random":
        return "processed_QH9Stable_random_12.json"
    if split == "size_ood":
        return "processed_QH9Stable_size_ood.json"
    raise ValueError(f"Unsupported QH9Stable split: {split}")


def make_random_split_indices(total_len: int, seed: int = 43) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_ratio = [0.8, 0.1, 0.1]
    train_len = int(total_len * data_ratio[0])
    val_len = int(total_len * data_ratio[1])
    indices = np.random.RandomState(seed=seed).permutation(total_len)
    train_mask = indices[:train_len].astype(np.int64)
    val_mask = indices[train_len : train_len + val_len].astype(np.int64)
    test_mask = indices[train_len + val_len :].astype(np.int64)
    return train_mask, val_mask, test_mask


def make_size_ood_split_indices(num_nodes: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes_array = np.asarray(list(num_nodes), dtype=np.int64)
    train_mask = np.where(num_nodes_array <= 20)[0].astype(np.int64)
    val_mask = np.where((num_nodes_array >= 21) & (num_nodes_array <= 22))[0].astype(np.int64)
    test_mask = np.where(num_nodes_array >= 23)[0].astype(np.int64)
    return train_mask, val_mask, test_mask


def dense_index_entries(num_items: int) -> list[tuple[int, int, int]]:
    return [(0, dense_idx, dense_idx) for dense_idx in range(num_items)]


def default_output_folder(root: Path, dataset_name: str) -> Path:
    return root / f"{dataset_name}_val_only"


def resolve_raw_db(root: Path, dataset_name: str, raw_db: str | None) -> Path:
    if raw_db:
        path = Path(raw_db).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        root / dataset_name / "raw" / f"{dataset_name}.db",
        root / f"{dataset_name}_shard" / "raw" / f"{dataset_name}.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find raw QH9 DB. Tried: "
        + ", ".join(str(path) for path in candidates)
        + ". Pass --raw-db explicitly."
    )


def first_table_name(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1").fetchone()
    if row is None:
        raise ValueError("No table found in raw database")
    return str(row[0])


def second_column_name(conn: sqlite3.Connection, table_name: str) -> str:
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    if len(columns) < 2:
        raise ValueError(f"Table {table_name} does not have the expected QH9 columns")
    return str(columns[1][1])


def load_split_masks(raw_db: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with sqlite3.connect(str(raw_db)) as conn:
        table_name = first_table_name(conn)
        total_len = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        if split == "random":
            return make_random_split_indices(total_len)
        if split == "size_ood":
            num_nodes_col = second_column_name(conn, table_name)
            num_nodes = [int(row[0]) for row in conn.execute(f"SELECT {num_nodes_col} FROM {table_name}")]
            return make_size_ood_split_indices(num_nodes)
    raise ValueError(f"Unsupported split: {split}")


def fetch_row_by_offset(conn: sqlite3.Connection, table_name: str, offset: int) -> tuple:
    row = conn.execute(f"SELECT * FROM {table_name} LIMIT 1 OFFSET ?", (int(offset),)).fetchone()
    if row is None:
        raise IndexError(f"Raw row offset {offset} not found in {table_name}")
    return row


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_output_dirs(processed_dir: Path, overwrite: bool) -> Path:
    if processed_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{processed_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(processed_dir)
    shard_dir = processed_dir / "lmdbs" / "shard_000.lmdb"
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    return shard_dir


def preprocess_validation_only(args: argparse.Namespace) -> None:
    import lmdb
    from tqdm.rich import tqdm

    from dataset_module.qh9_datasets_shard import QH9Stable_shard

    root = Path(args.root).expanduser().resolve()
    raw_db = resolve_raw_db(root, args.dataset_name, args.raw_db)
    output_folder = Path(args.output_folder).expanduser().resolve() if args.output_folder else default_output_folder(root, args.dataset_name)
    processed_dir = output_folder / "processed"

    train_mask, val_mask, test_mask = load_split_masks(raw_db, args.split)
    if args.max_val_samples is not None:
        val_mask = val_mask[: args.max_val_samples]

    if args.dry_run:
        print(f"RAW_DB {raw_db}")
        print(f"OUTPUT_FOLDER {output_folder}")
        print(f"SPLIT {args.split}")
        print(f"TRAIN_COUNT {len(train_mask)}")
        print(f"VAL_COUNT {len(val_mask)}")
        print(f"TEST_COUNT {len(test_mask)}")
        print("DRY_RUN true")
        return

    shard_dir = prepare_output_dirs(processed_dir, args.overwrite)
    in_process_dir = Path(str(shard_dir) + "_in_process")
    if in_process_dir.exists():
        shutil.rmtree(in_process_dir)

    processor = QH9Stable_shard(
        root_path=str(raw_db),
        shard_num=1,
        save_path=str(output_folder),
        use_parallel=False,
        make_split_info=False,
    )

    map_size = max(int(args.map_size_gb * (1024**3)), max(1, len(val_mask)) * 30 * 1024 * 1024 * 3)
    env = lmdb.open(str(in_process_dir), map_size=map_size)
    original_indices = [int(idx) for idx in val_mask.tolist()]
    with sqlite3.connect(str(raw_db)) as conn:
        table_name = first_table_name(conn)
        with env.begin(write=True) as txn:
            for dense_idx, original_idx in enumerate(tqdm(original_indices, desc="Processing validation rows")):
                row = fetch_row_by_offset(conn, table_name, original_idx)
                key, value = processor.process_data((row, dense_idx))
                txn.put(key, value)
    env.close()
    os.rename(in_process_dir, shard_dir)

    dense_val = list(range(len(original_indices)))
    split_payload = {
        "train": [],
        "val": dense_val,
        "test": [],
        "source_split": args.split,
        "source_dataset": args.dataset_name,
        "is_validation_only": True,
    }
    write_json(processed_dir / split_file_name(args.dataset_name, args.split), split_payload)
    write_json(processed_dir / "index.json", {"index": dense_index_entries(len(original_indices))})
    write_json(shard_dir / "single_index.json", {"index": dense_index_entries(len(original_indices))})
    write_json(processed_dir / "val_original_indices.json", {"indices": original_indices})
    write_json(
        processed_dir / "shard_completion_status.json",
        {
            "total_shards": 1,
            "completed_shards": [0],
            "missing_shards": [],
            "all_completed": True,
        },
    )
    (processed_dir / "ALL_SHARDS_COMPLETED.txt").write_text("validation-only preprocessing complete\n", encoding="utf-8")
    (processed_dir / "db_info.txt").write_text(
        f"source_raw_db: {raw_db}\nsource_split: {args.split}\nvalidation_rows: {len(original_indices)}\n",
        encoding="utf-8",
    )
    write_json(
        processed_dir / "validation_only_manifest.json",
        {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "raw_db": str(raw_db),
            "output_folder": str(output_folder),
            "processed_dir": str(processed_dir),
            "shard_num": 1,
            "prefix_for_loader": output_folder.name.removeprefix(args.dataset_name),
            "val_count": len(original_indices),
            "key_space": "dense_validation_indices",
            "original_indices_file": "val_original_indices.json",
            "load_example": (
                "QH9Stable(root='<dataset_root>', split='"
                + args.split
                + "', prefix='"
                + output_folder.name.removeprefix(args.dataset_name)
                + "', shard_num=1)"
            ),
        },
    )

    print("VAL_ONLY_PREPROCESS_OK")
    print(f"RAW_DB {raw_db}")
    print(f"OUTPUT_FOLDER {output_folder}")
    print(f"PROCESSED_DIR {processed_dir}")
    print(f"VAL_COUNT {len(original_indices)}")
    print(f"SHARD_DIR {shard_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess only QH9 validation rows into a val-only LMDB.")
    parser.add_argument("--root", default=str(SRC_ROOT.parent / "dataset"), help="QHFlow2 dataset root")
    parser.add_argument("--dataset-name", default="QH9Stable", choices=["QH9Stable"])
    parser.add_argument("--split", default="random", choices=["random", "size_ood"])
    parser.add_argument("--raw-db", default=None, help="Path to QH9Stable.db. Auto-detected from --root when omitted.")
    parser.add_argument("--output-folder", default=None, help="Output dataset folder. Default: <root>/QH9Stable_val_only")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for quick smoke preprocessing")
    parser.add_argument("--map-size-gb", type=float, default=16.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_validation_only(parse_args())
