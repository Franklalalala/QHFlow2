from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.dptb_compatible_monitor import (
    compute_dptb_compatible_component_losses,
    dptb_component_log_specs,
    required_dptb_sample_steps,
    unique_sample_metric_steps,
    validation_sample_metric_steps,
)


def test_component_losses_follow_dptb_l1_rmse_formula_with_masks():
    outputs = {
        "hamiltonian_diagonal_blocks": torch.tensor(
            [
                [[3.0, 999.0], [1.0, 5.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        ),
        "hamiltonian_non_diagonal_blocks": torch.tensor(
            [
                [[1.0, 3.0], [999.0, 999.0]],
                [[2.0, 999.0], [999.0, 6.0]],
            ]
        ),
    }
    target = SimpleNamespace(
        diagonal_hamiltonian=torch.zeros(2, 2, 2),
        diagonal_hamiltonian_mask=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ]
        ),
        non_diagonal_hamiltonian=torch.zeros(2, 2, 2),
        non_diagonal_hamiltonian_mask=torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )

    metrics = compute_dptb_compatible_component_losses(outputs, target)

    onsite_l1 = torch.tensor(13.0 / 5.0)
    onsite_mse = torch.tensor(43.0 / 5.0)
    hopping_l1 = torch.tensor(12.0 / 4.0)
    hopping_mse = torch.tensor(50.0 / 4.0)

    torch.testing.assert_close(metrics["onsite_mae"], onsite_l1)
    torch.testing.assert_close(metrics["onsite_mse"], onsite_mse)
    torch.testing.assert_close(metrics["onsite_loss"], 0.5 * (onsite_l1 + torch.sqrt(onsite_mse)))
    torch.testing.assert_close(metrics["hopping_mae"], hopping_l1)
    torch.testing.assert_close(metrics["hopping_mse"], hopping_mse)
    torch.testing.assert_close(metrics["hopping_loss"], 0.5 * (hopping_l1 + torch.sqrt(hopping_mse)))


def test_component_losses_skip_missing_hopping_blocks():
    outputs = {
        "hamiltonian_diagonal_blocks": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    }
    target = SimpleNamespace(
        diagonal_hamiltonian=torch.zeros(1, 2, 2),
        diagonal_hamiltonian_mask=torch.ones(1, 2, 2),
    )

    metrics = compute_dptb_compatible_component_losses(outputs, target)

    assert "onsite_loss" in metrics
    assert "hopping_loss" not in metrics


def test_component_losses_skip_non_block_outputs():
    outputs = {"hamiltonian": torch.zeros(1, 2, 2)}
    target = SimpleNamespace(hamiltonian=torch.zeros(1, 2, 2))

    metrics = compute_dptb_compatible_component_losses(outputs, target)

    assert metrics == {}


def test_component_losses_support_md17_water_dense_hamiltonian():
    outputs = {"hamiltonian": torch.ones(1, 24, 24)}
    target = SimpleNamespace(hamiltonian=torch.zeros(1, 24, 24))

    metrics = compute_dptb_compatible_component_losses(outputs, target)

    torch.testing.assert_close(metrics["onsite_loss"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["hopping_loss"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["onsite_count"], torch.tensor(246.0))
    torch.testing.assert_close(metrics["hopping_count"], torch.tensor(330.0))


def test_validation_sample_plan_keeps_only_one_euler_one_forward():
    assert unique_sample_metric_steps([1], 1) == [(1, "_1")]
    assert unique_sample_metric_steps([], 1) == [(1, "")]
    assert unique_sample_metric_steps([1, 3], 1) == [(1, "_1"), (3, "_3")]


def test_default_dptb_monitor_requires_only_euler_one_sample():
    assert required_dptb_sample_steps(True, False, [1, 3]) == [1]
    assert required_dptb_sample_steps(True, True, [1, 3]) == [1, 3]
    assert required_dptb_sample_steps(False, True, [1, 3]) == []


def test_test_sampling_plan_deduplicates_monitor_and_default_steps():
    steps = required_dptb_sample_steps(True, False, [1]) + [1]

    assert unique_sample_metric_steps(steps, 1) == [(1, "_1")]


def test_validation_plan_keeps_euler_one_legacy_sample_above_error_threshold():
    assert validation_sample_metric_steps(
        [3],
        3,
        threshold_passed=False,
        legacy_enabled=True,
    ) == [(1, "_1")]
    assert validation_sample_metric_steps(
        [3],
        3,
        threshold_passed=False,
        legacy_enabled=False,
    ) == []


def test_validation_plan_keeps_explicit_extra_steps_when_threshold_passes():
    steps = required_dptb_sample_steps(True, True, [1, 5]) + [3]

    assert validation_sample_metric_steps(
        steps,
        3,
        threshold_passed=True,
        legacy_enabled=True,
    ) == [(1, "_1"), (5, "_5"), (3, "_3")]


def test_validation_log_specs_match_deeptb_legacy_names_without_extra_tags():
    onsite_specs = dptb_component_log_specs(
        "val", "onsite_loss", 1, extra_tags=False
    )
    hopping_specs = dptb_component_log_specs(
        "val", "hopping_loss", 1, extra_tags=False
    )
    names = {spec["name"] for spec in onsite_specs + hopping_specs}

    assert "validation_onsite_loss_mean/epoch" in names
    assert "validation_hopping_loss_mean/epoch" in names
    assert not any("dptb_compatible" in name for name in names)
    assert not any("compatible_euler" in name for name in names)


def test_explicit_extra_tags_preserve_dptb_compatible_names():
    specs = dptb_component_log_specs("val", "onsite_loss", 1, extra_tags=True)
    names = {spec["name"] for spec in specs}

    assert "val/dptb_compatible_onsite_loss_euler1" in names
    assert "validation_compatible_euler_1_onsite_loss_mean/epoch" in names
