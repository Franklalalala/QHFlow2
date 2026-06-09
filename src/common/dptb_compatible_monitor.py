"""DPTB-compatible component losses for validation monitoring."""

from collections.abc import Mapping

import torch


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

    return metrics


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
