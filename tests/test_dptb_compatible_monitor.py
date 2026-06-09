from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.dptb_compatible_monitor import compute_dptb_compatible_component_losses
from pl_module.flow_module import LitModel_flow


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


def test_validation_logger_emits_dptb_style_aliases_for_euler_one():
    outputs, target, recorder = _make_logger_inputs()

    LitModel_flow._log_dptb_compatible_component_losses(recorder, outputs, target, "val", 1, "_1")

    record_by_name = {name: kwargs for name, _value, kwargs in recorder.records}
    assert "val_onsite_loss" in record_by_name
    assert "val_hopping_loss" in record_by_name
    assert "val/dptb_compatible_onsite_loss_euler1" in record_by_name
    assert "val/dptb_compatible_hopping_loss_euler1" in record_by_name
    assert record_by_name["val_onsite_loss"]["on_step"] is False
    assert record_by_name["val_onsite_loss"]["on_epoch"] is True
    assert record_by_name["val_hopping_loss"]["on_step"] is False
    assert record_by_name["val_hopping_loss"]["on_epoch"] is True


def test_test_logger_emits_dptb_style_aliases_for_euler_one():
    outputs, target, recorder = _make_logger_inputs()

    LitModel_flow._log_dptb_compatible_component_losses(recorder, outputs, target, "test", 1, "_1")

    record_by_name = {name: kwargs for name, _value, kwargs in recorder.records}
    assert "test_onsite_loss" in record_by_name
    assert "test_hopping_loss" in record_by_name
    assert "test/dptb_compatible_onsite_loss_euler1" in record_by_name
    assert "test/dptb_compatible_hopping_loss_euler1" in record_by_name
    assert record_by_name["test_onsite_loss"]["on_step"] is False
    assert record_by_name["test_onsite_loss"]["on_epoch"] is True
    assert record_by_name["test_hopping_loss"]["on_step"] is False
    assert record_by_name["test_hopping_loss"]["on_epoch"] is True


def _make_logger_inputs():
    class Recorder:
        dptb_compatible_monitor = True
        dptb_compatible_monitor_steps = [1]
        cur_batch_size = 1

        def __init__(self):
            self.records = []

        def log(self, name, value, **kwargs):
            self.records.append((name, value, kwargs))

    outputs = {
        "hamiltonian_diagonal_blocks": torch.ones(1, 2, 2),
        "hamiltonian_non_diagonal_blocks": torch.ones(1, 2, 2),
    }
    target = SimpleNamespace(
        diagonal_hamiltonian=torch.zeros(1, 2, 2),
        diagonal_hamiltonian_mask=torch.ones(1, 2, 2),
        non_diagonal_hamiltonian=torch.zeros(1, 2, 2),
        non_diagonal_hamiltonian_mask=torch.ones(1, 2, 2),
    )
    return outputs, target, Recorder()
