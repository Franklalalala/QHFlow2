from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import torch


def resolve_profile_options(options: Optional[Mapping] = None) -> Dict[str, object]:
    """Resolve conservative/aggressive Pixel MeanFlow defaults.

    Conservative is the default paper-semantic path for Hamiltonians: use the
    boundary/marginal velocity proxy as the JVP tangent and keep extra
    normalization or boundary-v losses off.  Aggressive is explicit opt-in.
    """
    options = dict(options or {})
    profile = str(options.get("profile", "conservative")).lower()
    if bool(options.get("aggressive", False)):
        profile = "aggressive"
    if profile not in {"conservative", "aggressive"}:
        raise ValueError("pixel_meanflow.profile must be conservative or aggressive")
    aggressive = profile == "aggressive"
    resolved = {
        "profile": profile,
        "time_sampling": options.get("time_sampling", "logit_normal"),
        "p_mean": float(options.get("p_mean", -0.4)),
        "p_std": float(options.get("p_std", 1.0)),
        "data_proportion": float(options.get("data_proportion", 0.50)),
        "tr_uniform_prob": float(options.get("tr_uniform_prob", 0.10)),
        "min_t": float(options.get("min_t", 0.05)),
        "fd_eps": float(options.get("fd_eps", 1.0e-3)),
        "jvp_backend": str(options.get("jvp_backend", "auto")).lower(),
        "jvp_create_graph": bool(options.get("jvp_create_graph", True)),
        "jvp_tangent": str(options.get("jvp_tangent", "boundary")).lower(),
        "aux_endpoint_weight": float(options.get("aux_endpoint_weight", 0.05)),
        "aux_boundary_v_weight": float(options.get("aux_boundary_v_weight", 0.10 if aggressive else 0.0)),
        "norm_p": float(options.get("norm_p", 1.0 if aggressive else 0.0)),
        "norm_eps": float(options.get("norm_eps", 0.01)),
        "time_conditioning": str(options.get("time_conditioning", "h" if aggressive else "trh")).lower(),
    }
    if resolved["jvp_tangent"] not in {"path", "boundary"}:
        raise ValueError("pixel_meanflow.jvp_tangent must be path or boundary")
    if resolved["jvp_backend"] not in {"auto", "autograd", "finite_difference"}:
        raise ValueError("pixel_meanflow.jvp_backend must be auto, autograd, or finite_difference")
    if resolved["time_conditioning"] not in {"t", "h", "trh"}:
        raise ValueError("pixel_meanflow.time_conditioning must be t, h, or trh")
    return resolved


def _batch_value(batch, key: str):
    if isinstance(batch, Mapping):
        return batch[key]
    try:
        return batch[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(batch, key)


def extract_endpoint_prediction(
    outputs: Mapping[str, torch.Tensor],
    batch,
    *,
    qh9: bool,
    use_res_target: bool,
    use_init_hamiltonian_residue: bool,
) -> torch.Tensor:
    if qh9:
        x_pred = outputs["hamiltonian_diagonal_blocks"]
        if use_res_target and use_init_hamiltonian_residue:
            x_pred = x_pred - _batch_value(batch, "diagonal_init_ham")
        return x_pred

    x_pred = outputs["hamiltonian"]
    if use_res_target and use_init_hamiltonian_residue:
        x_pred = x_pred - _batch_value(batch, "init_ham")
    return x_pred


def add_qh9_nondiag_endpoint_loss(
    errors: Dict[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    batch,
    *,
    weight: float,
    norm_eps: float,
) -> None:
    if weight <= 0.0 or "hamiltonian_non_diagonal_blocks" not in outputs:
        return

    off_pred = outputs["hamiltonian_non_diagonal_blocks"]
    off_tgt = _batch_value(batch, "non_diagonal_hamiltonian").to(
        device=off_pred.device, dtype=off_pred.dtype
    )
    off_mask = _batch_value(batch, "non_diagonal_hamiltonian_mask").to(
        device=off_pred.device, dtype=off_pred.dtype
    )
    off_loss, off_mse, off_mae = adaptive_masked_loss(
        off_pred - off_tgt,
        off_mask,
        norm_p=0.0,
        norm_eps=norm_eps,
    )
    errors["meanflow_nondiag_endpoint"] = off_loss
    errors["meanflow_nondiag_endpoint_mse"] = off_mse
    errors["meanflow_nondiag_endpoint_mae"] = off_mae
    errors["loss"] = errors["loss"] + weight * off_loss


def format_qh9_sample_result(
    H_t: torch.Tensor,
    outputs: Optional[Mapping[str, torch.Tensor]],
    *,
    use_non_diagonal_hamiltonian_scale: bool,
    non_diagonal_hamiltonian_scale: float,
) -> Dict[str, torch.Tensor]:
    result = {"hamiltonian_diagonal_blocks": H_t}
    if outputs is None:
        return result

    if "hamiltonian_non_diagonal_blocks" in outputs:
        non_diag = outputs["hamiltonian_non_diagonal_blocks"]
        if use_non_diagonal_hamiltonian_scale:
            non_diag = non_diag / non_diagonal_hamiltonian_scale
        result["hamiltonian_non_diagonal_blocks"] = non_diag

    for key in [
        "node_attr",
        "node_attr_init",
        "fii",
        "fij",
        "full_edge_index",
        "full_edge_distance_vec",
    ]:
        if key in outputs:
            result[key] = outputs[key]
    return result


def sample_two_times(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    min_t: float = 0.05,
    time_sampling: str = "logit_normal",
    p_mean: float = -0.4,
    p_std: float = 1.0,
    data_proportion: float = 0.50,
    tr_uniform_prob: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def sample_base() -> torch.Tensor:
        if time_sampling == "uniform":
            return torch.rand(n, device=device, dtype=dtype)
        if time_sampling == "logit_normal":
            raw = torch.randn(n, device=device, dtype=dtype) * p_std + p_mean
            return torch.sigmoid(raw)
        raise ValueError(f"Unsupported pixel_meanflow.time_sampling={time_sampling!r}")

    t = sample_base()
    r = sample_base()
    if tr_uniform_prob > 0.0:
        use_uniform = torch.rand(n, device=device) < tr_uniform_prob
        t = torch.where(use_uniform, torch.rand(n, device=device, dtype=dtype), t)
        r = torch.where(use_uniform, torch.rand(n, device=device, dtype=dtype), r)
    fm_mask = torch.rand(n, device=device) < data_proportion
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    t = t.clamp(min=min_t, max=1.0)
    r = torch.minimum(r.clamp(min=0.0, max=1.0), t)
    r = torch.where(fm_mask, t, r)
    return r, t, fm_mask


def time_view(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.reshape((-1,) + (1,) * (like.ndim - 1)).clamp_min(1.0e-8)


def average_velocity_from_endpoint(z_t: torch.Tensor, x_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (z_t - x_pred) / time_view(t, z_t)


def compound_velocity(
    u: torch.Tensor,
    du_dt: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    h_view = (t - r).reshape((-1,) + (1,) * (u.ndim - 1))
    return u + h_view * du_dt


def adaptive_masked_loss(
    diff: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    norm_p: float = 0.0,
    norm_eps: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask is None:
        reduce_dims = tuple(range(1, diff.ndim))
        per_mse = diff.square().mean(dim=reduce_dims) if reduce_dims else diff.square()
        per_mae = diff.abs().mean(dim=reduce_dims) if reduce_dims else diff.abs()
    else:
        mask_f = mask.to(device=diff.device, dtype=diff.dtype)
        if mask_f.shape != diff.shape:
            mask_f = mask_f.expand_as(diff)
        reduce_dims = tuple(range(1, diff.ndim))
        denom = mask_f.sum(dim=reduce_dims).clamp_min(1.0)
        per_mse = (diff.square() * mask_f).sum(dim=reduce_dims) / denom
        per_mae = (diff.abs() * mask_f).sum(dim=reduce_dims) / denom
    per_loss = per_mse
    if norm_p != 0.0:
        per_loss = per_loss / (per_loss.detach() + norm_eps).pow(norm_p)
    return per_loss.mean(), per_mse.mean(), per_mae.mean()


def meanflow_time_channels(data_dict: Mapping[str, torch.Tensor], mode: str = "trh") -> List[torch.Tensor]:
    mode = str(mode).lower()
    t = data_dict["t"]
    r = data_dict.get("r", None)
    h = data_dict.get("meanflow_h", None)
    if h is None and r is not None:
        h = (t - r).clamp_min(0.0)
    if mode == "t" or r is None:
        return [t]
    if mode == "h":
        return [h]
    if mode == "trh":
        return [t, r.to(device=t.device, dtype=t.dtype), h.to(device=t.device, dtype=t.dtype)]
    raise ValueError("time conditioning mode must be t, h, or trh")
