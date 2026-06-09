from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def normalize_subset(subset: str) -> str:
    subset = subset.lower()
    if subset in {"val", "valid", "validation"}:
        return "val"
    if subset == "test":
        return "test"
    raise ValueError(f"Unsupported subset: {subset}")


def default_output_prefix(subset: str) -> str:
    return "_valid_only_shard" if normalize_subset(subset) == "val" else "_test_only_shard"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_source_index(source_processed: Path) -> list[list[int]]:
    payload = load_json(source_processed / "index.json")
    entries = payload["index"] if isinstance(payload, dict) else payload
    return [[int(v) for v in entry] for entry in entries]


def split_payload_to_indices(payload: object, subset: str) -> list[int] | None:
    if not isinstance(payload, dict):
        return None
    keys = ["val", "valid", "validation"] if subset == "val" else ["test"]
    for key in keys:
        if key in payload:
            return [int(v) for v in payload[key]]
    return None


def make_random_split(total_len: int, num_train: int, num_valid: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = np.random.RandomState(seed=seed).permutation(total_len).astype(np.int64).tolist()
    train = indices[:num_train]
    val = indices[num_train : num_train + num_valid]
    test = indices[num_train + num_valid :]
    return train, val, test


def select_source_indices(
    source_processed: Path,
    subset: str,
    total_len: int,
    num_train: int,
    num_valid: int,
    split_seed: int,
    use_source_split: bool,
) -> tuple[list[int], str]:
    split_path = source_processed / "random_split.json"
    if use_source_split and split_path.exists():
        indices = split_payload_to_indices(load_json(split_path), subset)
        if indices is not None:
            return indices, str(split_path)

    _train, val, test = make_random_split(total_len, num_train, num_valid, split_seed)
    return (val if subset == "val" else test), f"generated_seed_{split_seed}"


def copy_small_sidecars(source_processed: Path, output_processed: Path) -> None:
    for name in ("pre_filter.pt", "pre_transform.pt"):
        source = source_processed / name
        if source.exists():
            shutil.copy2(source, output_processed / name)


def prepare_output(output_processed: Path, overwrite: bool) -> Path:
    if output_processed.exists():
        if not overwrite:
            raise FileExistsError(f"{output_processed} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_processed)
    shard_dir = output_processed / "lmdbs" / "shard_000.lmdb"
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    return shard_dir


def source_key_candidates(source_index: list[list[int]], source_idx: int) -> list[bytes]:
    candidates = [int(source_idx)]
    if 0 <= source_idx < len(source_index):
        _shard_idx, cur_idx, shard_data_idx = source_index[source_idx]
        candidates.extend([int(cur_idx), int(shard_data_idx)])
    seen = []
    for value in candidates:
        key = int(value).to_bytes(length=4, byteorder="big")
        if key not in seen:
            seen.append(key)
    return seen


def copy_subset(args: argparse.Namespace) -> None:
    import lmdb
    from tqdm.rich import tqdm

    subset = normalize_subset(args.subset)
    root = Path(args.root).expanduser().resolve()
    source_folder = root / f"{args.dataset_name}{args.source_prefix}"
    source_processed = source_folder / "processed"
    source_lmdb_dir = source_processed / "lmdbs"
    output_prefix = args.output_prefix or default_output_prefix(subset)
    output_folder = root / f"{args.dataset_name}{output_prefix}"
    output_processed = output_folder / "processed"

    source_index = load_source_index(source_processed)
    source_indices, split_source = select_source_indices(
        source_processed,
        subset,
        total_len=len(source_index),
        num_train=args.num_train,
        num_valid=args.num_valid,
        split_seed=args.split_seed,
        use_source_split=args.use_source_split,
    )
    if args.max_samples is not None:
        source_indices = source_indices[: args.max_samples]

    if args.dry_run:
        print(f"SOURCE_FOLDER {source_folder}")
        print(f"OUTPUT_FOLDER {output_folder}")
        print(f"SUBSET {subset}")
        print(f"SPLIT_SOURCE {split_source}")
        print(f"SELECTED_COUNT {len(source_indices)}")
        print("DRY_RUN true")
        return

    shard_dir = prepare_output(output_processed, args.overwrite)
    in_process_dir = Path(str(shard_dir) + "_in_process")
    if in_process_dir.exists():
        shutil.rmtree(in_process_dir)

    map_size = max(int(args.map_size_gb * (1024**3)), max(1, len(source_indices)) * 3 * 1024 * 1024)
    out_env = lmdb.open(str(in_process_dir), map_size=map_size)
    source_envs = {}
    try:
        with out_env.begin(write=True) as out_txn:
            for dense_idx, source_idx in enumerate(tqdm(source_indices, desc=f"Copying MD17 water {subset}")):
                if not 0 <= source_idx < len(source_index):
                    raise IndexError(f"Source index {source_idx} outside index.json range")
                shard_idx = int(source_index[source_idx][0])
                if shard_idx not in source_envs:
                    source_envs[shard_idx] = lmdb.open(
                        str(source_lmdb_dir / f"shard_{shard_idx:03d}.lmdb"),
                        readonly=True,
                        lock=False,
                        readahead=False,
                    )
                value = None
                with source_envs[shard_idx].begin() as in_txn:
                    for key in source_key_candidates(source_index, source_idx):
                        value = in_txn.get(key)
                        if value is not None:
                            break
                if value is None:
                    raise KeyError(f"Could not find source index {source_idx} in shard {shard_idx}")
                if args.validate_pickle:
                    pickle.loads(value)
                out_key = int(dense_idx).to_bytes(length=4, byteorder="big")
                out_txn.put(out_key, value)
    finally:
        out_env.close()
        for env in source_envs.values():
            env.close()

    os.rename(in_process_dir, shard_dir)
    copy_small_sidecars(source_processed, output_processed)

    dense_indices = list(range(len(source_indices)))
    split_payload = {
        "train": [],
        "val": dense_indices if subset == "val" else [],
        "test": dense_indices if subset == "test" else [],
        "source_dataset": args.dataset_name,
        "source_prefix": args.source_prefix,
        "source_split": split_source,
        "subset": subset,
        "use_source_split": args.use_source_split,
    }
    write_json(output_processed / "random_split.json", split_payload)
    write_json(output_processed / "index.json", {"index": [[0, idx, idx] for idx in dense_indices]})
    write_json(shard_dir / "single_index.json", {"index": [[0, idx, idx] for idx in dense_indices]})
    write_json(output_processed / f"{subset}_source_indices.json", {"indices": source_indices})
    write_json(
        output_processed / "shard_completion_status.json",
        {"total_shards": 1, "completed_shards": [0], "missing_shards": [], "all_completed": True},
    )
    (output_processed / "ALL_SHARDS_COMPLETED.txt").write_text(
        f"MD17 water {subset}-only shard copied from preprocessed data\n", encoding="utf-8"
    )
    (output_processed / "db_info.txt").write_text(
        "\n".join(
            [
                f"source_folder: {source_folder}",
                f"source_split: {split_source}",
                f"subset: {subset}",
                f"subset_rows: {len(source_indices)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        output_processed / "subset_only_manifest.json",
        {
            "dataset_name": args.dataset_name,
            "source_folder": str(source_folder),
            "output_folder": str(output_folder),
            "output_prefix": output_prefix,
            "subset": subset,
            "subset_count": len(source_indices),
            "shard_num": 1,
            "copied_from_preprocessed_lmdb": True,
            "compute_q_tensor_required": False,
            "use_source_split": args.use_source_split,
            "load_example": (
                "MD17_DFT_Shard(root='<repo>/dataset', name='"
                + args.dataset_name
                + "', prefix='"
                + output_prefix
                + "', shard_num=1, compute_q_tensor=False)"
            ),
        },
    )

    print(f"MD17_WATER_{subset.upper()}_ONLY_PREPROCESS_OK")
    print(f"SOURCE_FOLDER {source_folder}")
    print(f"OUTPUT_FOLDER {output_folder}")
    print(f"PROCESSED_DIR {output_processed}")
    print(f"SELECTED_COUNT {len(source_indices)}")
    print(f"SHARD_DIR {shard_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy preprocessed MD17 water split rows into a subset-only shard.")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--dataset-name", default="water")
    parser.add_argument("--source-prefix", default="_shard")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--subset", choices=["val", "valid", "validation", "test"], default="val")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-valid", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--use-source-split", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-pickle", action="store_true")
    return parser


def main() -> None:
    copy_subset(build_parser().parse_args())


if __name__ == "__main__":
    main()
