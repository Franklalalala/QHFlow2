from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import common.pixel_meanflow as pmf
from common.pixel_meanflow import (
    average_velocity_from_endpoint,
    compound_velocity,
    meanflow_time_channels,
    resolve_profile_options,
    sample_two_times,
)


def test_sample_two_times_respects_order_and_fm_boundary():
    torch.manual_seed(20260613)

    r, t, fm_mask = sample_two_times(
        128,
        device=torch.device("cpu"),
        min_t=0.05,
        time_sampling="logit_normal",
        data_proportion=0.5,
        tr_uniform_prob=0.1,
    )

    assert r.shape == t.shape == fm_mask.shape == (128,)
    assert torch.all(r <= t)
    assert torch.all(t >= 0.05)
    assert torch.all(r[fm_mask] == t[fm_mask])
    assert 0.25 < float(fm_mask.float().mean()) < 0.75


def test_endpoint_parameterization_matches_paper_average_velocity_oracle():
    torch.manual_seed(20260613)
    x = torch.randn(16, 3, 3)
    eps = torch.randn(16, 3, 3)
    t = torch.full((16,), 0.6, requires_grad=True)
    r = torch.full((16,), 0.2, requires_grad=True)
    z_t = ((1.0 - t).reshape(-1, 1, 1) * x + t.reshape(-1, 1, 1) * eps).detach().requires_grad_(True)
    target_v = eps - x

    def u_fn(z_in, t_in, r_in):
        del r_in
        return average_velocity_from_endpoint(z_in, x, t_in)

    u = u_fn(z_t, t, r)
    _, du_dt = torch.autograd.functional.jvp(
        u_fn,
        (z_t, t, r),
        (target_v, torch.ones_like(t), torch.zeros_like(r)),
        strict=False,
    )
    v_theta = compound_velocity(u, du_dt.detach(), r, t)

    torch.testing.assert_close(u, target_v, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(v_theta, target_v, atol=1.0e-6, rtol=1.0e-6)


def test_meanflow_time_channels_support_h_only_and_trh_modes():
    data = {
        "t": torch.tensor([0.8, 0.8]),
        "r": torch.tensor([0.2, 0.6]),
        "meanflow_h": torch.tensor([0.6, 0.2]),
    }

    h_only = meanflow_time_channels(data, mode="h")
    trh = meanflow_time_channels(data, mode="trh")

    assert len(h_only) == 1
    assert torch.equal(h_only[0], torch.tensor([0.6, 0.2]))
    assert len(trh) == 3
    assert torch.equal(trh[0], data["t"])
    assert torch.equal(trh[1], data["r"])
    assert torch.equal(trh[2], data["meanflow_h"])


def test_aggressive_profile_is_explicit_opt_in():
    conservative = resolve_profile_options({})
    aggressive = resolve_profile_options({"profile": "aggressive"})

    assert conservative["profile"] == "conservative"
    assert conservative["jvp_tangent"] == "boundary"
    assert conservative["norm_p"] == 0.0
    assert conservative["aux_boundary_v_weight"] == 0.0

    assert aggressive["profile"] == "aggressive"
    assert aggressive["jvp_tangent"] == "boundary"
    assert aggressive["norm_p"] == 1.0
    assert aggressive["aux_boundary_v_weight"] > 0.0
    assert aggressive["time_conditioning"] == "h"


def test_extract_x_pred_subtracts_init_only_when_forward_adds_init():
    outputs = {"hamiltonian_diagonal_blocks": torch.tensor([[[5.0]]])}
    batch = {"diagonal_init_ham": torch.tensor([[[2.0]]])}

    torch.testing.assert_close(
        pmf.extract_endpoint_prediction(
            outputs,
            batch,
            qh9=True,
            use_res_target=True,
            use_init_hamiltonian_residue=False,
        ),
        torch.tensor([[[5.0]]]),
    )
    torch.testing.assert_close(
        pmf.extract_endpoint_prediction(
            outputs,
            batch,
            qh9=True,
            use_res_target=True,
            use_init_hamiltonian_residue=True,
        ),
        torch.tensor([[[3.0]]]),
    )

    outputs = {"hamiltonian": torch.tensor([[[7.0]]])}
    batch = SimpleNamespace(init_ham=torch.tensor([[[3.0]]]))
    torch.testing.assert_close(
        pmf.extract_endpoint_prediction(
            outputs,
            batch,
            qh9=False,
            use_res_target=True,
            use_init_hamiltonian_residue=False,
        ),
        torch.tensor([[[7.0]]]),
    )
    torch.testing.assert_close(
        pmf.extract_endpoint_prediction(
            outputs,
            batch,
            qh9=False,
            use_res_target=True,
            use_init_hamiltonian_residue=True,
        ),
        torch.tensor([[[4.0]]]),
    )


def test_qh9_nondiag_endpoint_aux_adds_loss_and_metrics():
    errors = {"loss": torch.tensor(1.0)}
    outputs = {"hamiltonian_non_diagonal_blocks": torch.tensor([[3.0, 5.0]])}
    batch = {
        "non_diagonal_hamiltonian": torch.tensor([[1.0, 1.0]]),
        "non_diagonal_hamiltonian_mask": torch.tensor([[1.0, 0.0]]),
    }

    pmf.add_qh9_nondiag_endpoint_loss(errors, outputs, batch, weight=2.0, norm_eps=0.01)

    assert errors["meanflow_nondiag_endpoint"].item() == pytest.approx(4.0)
    assert errors["meanflow_nondiag_endpoint_mse"].item() == pytest.approx(4.0)
    assert errors["meanflow_nondiag_endpoint_mae"].item() == pytest.approx(2.0)
    assert errors["loss"].item() == pytest.approx(9.0)


def test_qh9_nondiag_endpoint_aux_matches_scaled_original_criterion_space():
    errors = {"loss": torch.tensor(1.0)}
    outputs = {"hamiltonian_non_diagonal_blocks": torch.tensor([[6.0, 5.0]])}
    batch = {
        "non_diagonal_hamiltonian": torch.tensor([[3.0, 1.0]]),
        "non_diagonal_hamiltonian_mask": torch.tensor([[1.0, 0.0]]),
    }

    pmf.add_qh9_nondiag_endpoint_loss(
        errors,
        outputs,
        batch,
        weight=2.0,
        norm_eps=0.01,
        use_non_diagonal_hamiltonian_scale=True,
        non_diagonal_hamiltonian_scale=2.0,
    )

    assert errors["meanflow_nondiag_endpoint"].item() == pytest.approx(0.0)
    assert errors["meanflow_nondiag_endpoint_mse"].item() == pytest.approx(0.0)
    assert errors["meanflow_nondiag_endpoint_mae"].item() == pytest.approx(0.0)
    assert errors["loss"].item() == pytest.approx(1.0)


def test_qh9_pmf_sample_result_preserves_flow_module_contract():
    H_t = torch.tensor([[[11.0]]])
    outputs = {
        "hamiltonian_non_diagonal_blocks": torch.tensor([[8.0]]),
        "node_attr": torch.tensor([[1.0, 2.0]]),
        "node_attr_init": torch.tensor([[3.0, 4.0]]),
        "fii": torch.tensor([5.0]),
        "fij": torch.tensor([6.0]),
        "full_edge_index": torch.tensor([[0], [1]]),
        "full_edge_distance_vec": torch.tensor([[0.1, 0.2, 0.3]]),
    }

    result = pmf.format_qh9_sample_result(
        H_t,
        outputs,
        use_non_diagonal_hamiltonian_scale=True,
        non_diagonal_hamiltonian_scale=2.0,
    )

    torch.testing.assert_close(result["hamiltonian_diagonal_blocks"], H_t)
    torch.testing.assert_close(result["hamiltonian_non_diagonal_blocks"], torch.tensor([[4.0]]))
    for key in ["node_attr", "node_attr_init", "fii", "fij", "full_edge_index", "full_edge_distance_vec"]:
        assert result[key] is outputs[key]
