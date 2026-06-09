from pathlib import Path
import importlib.util

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataset_module"
    / "process"
    / "preprocess_qh9_validation_only.py"
)
SPEC = importlib.util.spec_from_file_location("preprocess_qh9_validation_only", SCRIPT)
preprocess = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocess)


def test_random_split_matches_qh9_stable_ratios_and_seed():
    train_mask, val_mask, test_mask = preprocess.make_random_split_indices(10)

    assert len(train_mask) == 8
    assert len(val_mask) == 1
    assert len(test_mask) == 1
    assert sorted(np.concatenate([train_mask, val_mask, test_mask]).tolist()) == list(range(10))

    expected = np.random.RandomState(seed=43).permutation(10)
    assert train_mask.tolist() == expected[:8].tolist()
    assert val_mask.tolist() == expected[8:9].tolist()
    assert test_mask.tolist() == expected[9:].tolist()


def test_size_ood_split_matches_qh9_stable_node_ranges():
    train_mask, val_mask, test_mask = preprocess.make_size_ood_split_indices([2, 20, 21, 22, 23, 30])

    assert train_mask.tolist() == [0, 1]
    assert val_mask.tolist() == [2, 3]
    assert test_mask.tolist() == [4, 5]


def test_validation_only_index_is_dense_for_loader_get():
    assert preprocess.dense_index_entries(3) == [(0, 0, 0), (0, 1, 1), (0, 2, 2)]


def test_split_file_name_is_loader_compatible():
    assert preprocess.split_file_name("QH9Stable", "random") == "processed_QH9Stable_random_12.json"
    assert preprocess.split_file_name("QH9Stable", "size_ood") == "processed_QH9Stable_size_ood.json"
