from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


def get_classifier_weight(model: nn.Module, expert: int) -> Optional[torch.Tensor]:
    """Return classifier weights as class prototypes when available."""
    if hasattr(model, "linear"):
        linear = model.linear
        if isinstance(linear, nn.ModuleList):
            return linear[expert].weight
        if isinstance(linear, nn.Linear):
            return linear.weight
    if hasattr(model, "fc"):
        return model.fc.weight
    if hasattr(model, "classifier"):
        return model.classifier.weight
    return None


def sample_from_buffer_by_class(buffer, target_class: int, n_samples: int, target_device=None):
    return buffer.sample_class_data(target_class, n_samples, target_device=target_device)


def sample_from_buffer_random_old(
    buffer,
    n_samples: int,
    old_classes: Iterable[int],
    target_device=None,
    exclude_indices=None,
):
    return buffer.sample_from_classes(
        old_classes, n_samples, target_device=target_device, exclude_indices=exclude_indices,
    )


def sample_hard_from_buffer(
    buffer,
    batch_size: int,
    exclude_task_id=None,
    target_device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(buffer) == 0 or batch_size <= 0:
        device = target_device if target_device else torch.device("cuda")
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long),
        )

    current_len = len(buffer)
    ref_device = buffer.bx.device if hasattr(buffer, "bx") else torch.device("cuda")

    buf_t = buffer.bt[:current_len] if hasattr(buffer, "bt") else torch.full((current_len,), -1, device=ref_device)
    buf_u = (
        buffer.uncertainty[:current_len]
        if hasattr(buffer, "uncertainty")
        else torch.zeros(current_len, device=ref_device)
    )

    if exclude_task_id is not None:
        valid_mask = (buf_t != exclude_task_id)
    else:
        valid_mask = torch.ones(current_len, dtype=torch.bool, device=ref_device)

    if not valid_mask.any():
        device = target_device if target_device else torch.device("cuda")
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long),
        )

    valid_indices = torch.nonzero(valid_mask).squeeze(1)
    valid_uncertainties = buf_u[valid_indices]
    weights = valid_uncertainties + 1e-6

    num_valid = valid_indices.numel()
    real_batch_size = min(batch_size, num_valid)
    if real_batch_size == 0:
        device = target_device if target_device else torch.device("cuda")
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long),
        )

    sampled_idx_in_valid = torch.multinomial(weights, real_batch_size, replacement=False)
    final_indices = valid_indices[sampled_idx_in_valid]

    ret_x = buffer.bx[final_indices]
    ret_y = buffer.by[final_indices]

    if target_device is not None:
        ret_x = ret_x.to(target_device, non_blocking=True)
        ret_y = ret_y.to(target_device, non_blocking=True)

    final_indices = final_indices.cpu().long()
    return ret_x, ret_y, final_indices
