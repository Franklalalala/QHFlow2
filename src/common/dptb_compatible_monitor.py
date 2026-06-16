"""DPTB-compatible component losses for validation monitoring."""

from collections.abc import Mapping

import torch


WATER_AO_SLICES = ((0, 14), (14, 19), (19, 24))


def unique_sample_metric_steps(log_steps, default_step):
    """Return a de-duplicated sampling plan with QHFlow2 suffix semantics."""
    plan = []
    seen = set()
    for n_steps in log_steps or []:
        if n_steps in seen:
            continue
        seen.add(n_steps)
        plan.append((n_steps, f"_{n_steps}"))
    if default_step not in seen:
        plan.append((default_step, ""))
    return plan


def required_dptb_sample_steps(legacy_enabled, extra_tags, monitor_steps):
    """Sampling steps needed only for DPTB-compatible TensorBoard logging.

    The default contract keeps Euler-1 DPTB-style validation/test component
    tags available. Extra tags request additional monitor steps and legacy
    QHFlow-style aliases.
    """
    if not legacy_enabled:
        return []
    required = [1]
    if extra_tags:
        for n_steps in monitor_steps or []:
            if n_steps not in required:
                required.append(n_steps)
    return required


def validation_sample_metric_steps(log_steps, default_step, *, threshold_passed, legacy_enabled):
    """Validation sampling plan.

    Normal sample metrics still obey the error threshold. Euler-1 legacy
    compatible aliases are kept outside the threshold gate so early validation
    epochs have the same DeePTB-comparable component tags.
    """
    if threshold_passed:
        steps = list(log_steps or [])
        if legacy_enabled and 1 not in steps:
            steps.insert(0, 1)
        return unique_sample_metric_steps(steps, default_step)
    if legacy_enabled:
        return unique_sample_metric_steps([1], 1)
    return []


def dptb_component_log_specs(prefix, key, num_timesteps, *, extra_tags=False):
    """Return TensorBoard log specs for a DPTB-compatible scalar."""
    specs = []
    is_component_loss = key in {"onsite_loss", "hopping_loss"}
    if extra_tags:
        specs.append(
            {
                "name": f"{prefix}/dptb_compatible_{key}_euler{num_timesteps}",
                "on_step": True,
                "on_epoch": True,
                "batch_size": None,
            }
        )
    if not is_component_loss:
        return specs

    canonical_prefix = "validation" if prefix == "val" else "test"
    if extra_tags:
        specs.append(
            {
                "name": f"{canonical_prefix}_compatible_euler_{num_timesteps}_{key}",
                "on_step": False,
                "on_epoch": True,
                "batch_size": 1,
            }
        )
    if int(num_timesteps) == 1:
        if extra_tags:
            specs.extend(
                [
                    {
                        "name": f"{prefix}/{key}",
                        "on_step": False,
                        "on_epoch": True,
                        "batch_size": None,
                    },
                    {
                        "name": f"{prefix}_{key}",
                        "on_step": False,
                        "on_epoch": True,
                        "batch_size": None,
                    },
                ]
            )
        specs.append(
            {
                "name": f"{canonical_prefix}_{key}",
                "on_step": False,
                "on_epoch": True,
                "batch_size": 1,
            }
        )
    return specs


def compute_dptb_compatible_component_losses(outputs, target, eps=1e-12):
    """Return onsite/hopping losses using the DPTB component formula.

    DPTB component curves use ``0.5 * (L1_mean + RMSE)`` over valid block
    elements. QHFlow names the same components diagonal and non-diagonal
    Hamiltonian blocks, so this helper maps them to onsite and hopping.
    """
    metrics = {}

    onsite = _component_loss(
        pred=_get_field(outputs, "hamiltonian_diagonal_blocks", required=False),
        target=_get_field(target, "diagonal_hamiltonian", required=False),
        mask=_get_field(target, "diagonal_hamiltonian_mask", required=False),
        eps=eps,
    )
    if onsite is not None:
        _add_prefixed(metrics, "onsite", onsite)

    hopping = _component_loss(
        pred=_get_field(outputs, "hamiltonian_non_diagonal_blocks", required=False),
        target=_get_field(target, "non_diagonal_hamiltonian", required=False),
        mask=_get_field(target, "non_diagonal_hamiltonian_mask", required=False),
        eps=eps,
    )
    if hopping is not None:
        _add_prefixed(metrics, "hopping", hopping)

    if not metrics:
        dense_metrics = _dense_water_component_losses(
            pred=_get_field(outputs, "hamiltonian", required=False),
            target=_get_field(target, "hamiltonian", required=False),
            eps=eps,
        )
        metrics.update(dense_metrics)

    return metrics


def _dense_water_component_losses(pred, target, eps=1e-12):
    pred = _as_batched_dense_matrix(pred)
    target = _as_batched_dense_matrix(target)
    if pred is None or target is None:
        return {}
    if pred.shape[-2:] != (24, 24) or target.shape[-2:] != (24, 24):
        return {}

    target = target.to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        return {}

    onsite_pred, onsite_target = _select_dense_blocks(pred, target, diagonal=True)
    hopping_pred, hopping_target = _select_dense_blocks(pred, target, diagonal=False)

    metrics = {}
    onsite = _component_loss(onsite_pred, onsite_target, eps=eps)
    if onsite is not None:
        _add_prefixed(metrics, "onsite", onsite)
    hopping = _component_loss(hopping_pred, hopping_target, eps=eps)
    if hopping is not None:
        _add_prefixed(metrics, "hopping", hopping)
    return metrics


def _as_batched_dense_matrix(value):
    if value is None or not torch.is_tensor(value):
        return None
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim == 3:
        return value
    if value.ndim == 4 and value.shape[1] == 1:
        return value[:, 0]
    return None


def _select_dense_blocks(pred, target, diagonal):
    pred_blocks = []
    target_blocks = []
    for i, (row_start, row_stop) in enumerate(WATER_AO_SLICES):
        for j, (col_start, col_stop) in enumerate(WATER_AO_SLICES):
            if diagonal != (i == j):
                continue
            pred_blocks.append(pred[:, row_start:row_stop, col_start:col_stop].reshape(pred.shape[0], -1))
            target_blocks.append(target[:, row_start:row_stop, col_start:col_stop].reshape(target.shape[0], -1))
    return torch.cat(pred_blocks, dim=1), torch.cat(target_blocks, dim=1)


def _component_loss(pred, target, mask=None, eps=1e-12):
    if pred is None or target is None:
        return None

    target = target.to(device=pred.device, dtype=pred.dtype)
    diff = pred - target

    if mask is None:
        count = torch.tensor(diff.numel(), device=diff.device, dtype=diff.dtype)
    else:
        mask = mask.to(device=diff.device, dtype=diff.dtype)
        diff = diff * mask
        count = mask.sum()

    if count.detach().item() <= 0:
        return None

    abs_sum = diff.abs().sum()
    square_sum = diff.pow(2).sum()
    mae = abs_sum / count
    mse = square_sum / count
    rmse = torch.sqrt(mse + eps)
    return {
        "loss": 0.5 * (mae + rmse),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "count": count.detach(),
    }


def _add_prefixed(metrics, prefix, values):
    for key, value in values.items():
        metrics[f"{prefix}_{key}"] = value


def _get_field(obj, key, required=True):
    value = None
    found = False

    if isinstance(obj, Mapping):
        if key in obj:
            value = obj[key]
            found = True
    else:
        try:
            value = obj[key]
            found = True
        except (KeyError, TypeError, AttributeError):
            if hasattr(obj, key):
                value = getattr(obj, key)
                found = True

    if found:
        return value
    if required:
        raise KeyError(key)
    return None
