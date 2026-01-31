from typing import Callable, Dict, Iterable, List, Optional, Tuple

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _as_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = x.t() @ y
    xtx = x.t() @ x
    yty = y.t() @ y
    num = (xty * xty).sum()
    denom = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum())
    if denom.item() == 0.0:
        return float("nan")
    return float((num / denom).item())


def _pool_feature(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    if feat.dim() == 3:
        return feat.mean(dim=1)
    return feat


def _flatten_feats(feats: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    flat = []
    for feat in feats:
        pooled = _pool_feature(feat)
        flat.append(pooled.reshape(pooled.size(0), -1))
    return flat


@torch.no_grad()
def layerwise_cka(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    *,
    device: Optional[torch.device] = None,
    max_batches: int = 10,
) -> List[float]:
    model_a.eval()
    model_b.eval()

    if device is None:
        device = next(model_a.parameters()).device

    feats_a: List[List[torch.Tensor]] = []
    feats_b: List[List[torch.Tensor]] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch[0].to(device, non_blocking=True)
        feats_a_raw = model_a.features(x) if hasattr(model_a, "features") else model_a(x)
        feats_b_raw = model_b.features(x) if hasattr(model_b, "features") else model_b(x)

        if not isinstance(feats_a_raw, (list, tuple)):
            feats_a_raw = [feats_a_raw]
        if not isinstance(feats_b_raw, (list, tuple)):
            feats_b_raw = [feats_b_raw]

        feats_a_raw = _flatten_feats(feats_a_raw)
        feats_b_raw = _flatten_feats(feats_b_raw)

        if not feats_a:
            feats_a = [[] for _ in range(len(feats_a_raw))]
            feats_b = [[] for _ in range(len(feats_b_raw))]

        for i, feat in enumerate(feats_a_raw):
            feats_a[i].append(feat.detach().cpu())
        for i, feat in enumerate(feats_b_raw):
            feats_b[i].append(feat.detach().cpu())

    scores = []
    for layer_idx in range(len(feats_a)):
        x = torch.cat(feats_a[layer_idx], dim=0)
        y = torch.cat(feats_b[layer_idx], dim=0)
        scores.append(_linear_cka(x, y))
    return scores


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_csv(path: str, rows: List[List[object]]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "cka_ours", "cka_baseline"])
        writer.writerows(rows)


def _plot_layerwise_cka(
    ours: List[float],
    baseline: Optional[List[float]],
    save_path: str,
    *,
    label_ours: str = "ATSD",
    label_baseline: str = "Baseline",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = np.arange(1, len(ours) + 1)
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(x, ours, marker="o", label=label_ours, color="#1b9e77")
    if baseline is not None:
        plt.plot(x, baseline, marker="s", label=label_baseline, color="#d95f02")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer")
    plt.ylabel("CKA Similarity")
    plt.title("Layer-wise Feature Drift (CKA)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_task_dataset(loader, save_path: str, *, max_batches: Optional[int] = None) -> None:
    _ensure_dir(os.path.dirname(save_path) or ".")
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        xs.append(batch[0].detach().cpu())
        ys.append(batch[1].detach().cpu())
    if xs:
        payload = {"x": torch.cat(xs, dim=0), "y": torch.cat(ys, dim=0)}
    else:
        payload = {
            "x": torch.empty((0,)),
            "y": torch.empty((0,), dtype=torch.long),
        }
    torch.save(payload, save_path)


def build_loader_from_tensor_file(
    path: str,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    payload = torch.load(path, map_location="cpu")
    dataset = TensorDataset(payload["x"], payload["y"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run_layerwise_cka_from_checkpoints(
    model_factory: Callable[[], torch.nn.Module],
    ckpt_task_path: str,
    ckpt_final_path: str,
    loader,
    *,
    device: Optional[torch.device] = None,
    max_batches: int = 10,
    out_dir: str = "./outputs/cka",
    run_name: str = "run",
    baseline_init_path: Optional[str] = None,
    baseline_final_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    task_ckpt = torch.load(ckpt_task_path, map_location="cpu")
    final_ckpt = torch.load(ckpt_final_path, map_location="cpu")
    task_state = _as_state_dict(task_ckpt)
    final_state = _as_state_dict(final_ckpt)
    return run_layerwise_cka_analysis(
        model_factory=model_factory,
        init_state=task_state,
        final_state=final_state,
        loader=loader,
        device=device,
        max_batches=max_batches,
        out_dir=out_dir,
        run_name=run_name,
        baseline_init_path=baseline_init_path,
        baseline_final_path=baseline_final_path,
    )


def run_layerwise_cka_analysis(
    model_factory: Callable[[], torch.nn.Module],
    init_state: Dict[str, torch.Tensor],
    final_state: Dict[str, torch.Tensor],
    loader,
    *,
    device: Optional[torch.device] = None,
    max_batches: int = 10,
    out_dir: str = "./outputs/cka",
    run_name: str = "run",
    baseline_init_path: Optional[str] = None,
    baseline_final_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    _ensure_dir(out_dir)
    out_dir = os.path.join(out_dir, run_name)
    _ensure_dir(out_dir)

    model_init = model_factory()
    model_final = model_factory()
    model_init.load_state_dict(init_state, strict=False)
    model_final.load_state_dict(final_state, strict=False)

    if device is not None:
        model_init = model_init.to(device)
        model_final = model_final.to(device)

    ours_scores = layerwise_cka(
        model_init,
        model_final,
        loader,
        device=device,
        max_batches=max_batches,
    )

    baseline_scores = None
    if baseline_init_path and baseline_final_path:
        base_init_ckpt = torch.load(baseline_init_path, map_location="cpu")
        base_final_ckpt = torch.load(baseline_final_path, map_location="cpu")
        base_init_state = _as_state_dict(base_init_ckpt)
        base_final_state = _as_state_dict(base_final_ckpt)

        base_init = model_factory()
        base_final = model_factory()
        base_init.load_state_dict(base_init_state, strict=False)
        base_final.load_state_dict(base_final_state, strict=False)
        if device is not None:
            base_init = base_init.to(device)
            base_final = base_final.to(device)

        baseline_scores = layerwise_cka(
            base_init,
            base_final,
            loader,
            device=device,
            max_batches=max_batches,
        )

    csv_rows = []
    for idx, ours in enumerate(ours_scores, start=1):
        base = baseline_scores[idx - 1] if baseline_scores is not None else ""
        csv_rows.append([idx, ours, base])
    _save_csv(os.path.join(out_dir, "cka_layerwise.csv"), csv_rows)

    _plot_layerwise_cka(
        ours_scores,
        baseline_scores,
        os.path.join(out_dir, "cka_layerwise.png"),
    )

    return {"ours": ours_scores, "baseline": baseline_scores or []}
