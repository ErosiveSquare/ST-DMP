import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import torch


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(12):
        if os.path.isfile(os.path.join(cur, "experiment", "dataset.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


def _parse_task_ids(value: str, n_tasks: int) -> List[int]:
    if value.strip().lower() in {"all", ""}:
        return list(range(n_tasks))
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _seed_dir_candidates(root: str, seed: int) -> List[str]:
    return [
        os.path.join(root, f"run_{seed:02d}"),
        os.path.join(root, f"run_{seed}"),
        os.path.join(root, f"seed_{seed}"),
    ]


def _resolve_ckpt_path(root: str, seed: int, task_id: int) -> str:
    tried = []
    for seed_dir in _seed_dir_candidates(root, seed):
        patterns = [
            os.path.join(seed_dir, f"task_{task_id}.pt"),
            os.path.join(seed_dir, f"task_{task_id}.pth"),
            os.path.join(seed_dir, f"ckpt_task_{task_id}.pth"),
            os.path.join(seed_dir, f"ckpt_task_{task_id}.pt"),
        ]
        tried.append(seed_dir)
        for path in patterns:
            if os.path.isfile(path):
                return path
        globbed = glob.glob(os.path.join(seed_dir, f"*task*{task_id}*"))
        if globbed:
            return sorted(globbed)[0]
    raise FileNotFoundError(
        f"Checkpoint not found for task {task_id}. Tried: {', '.join(tried)}"
    )


def _pick_task_loader(task_entry: dict, prefer_test: bool) -> torch.utils.data.DataLoader:
    if prefer_test and "test" in task_entry:
        return task_entry["test"]
    if "train" in task_entry:
        return task_entry["train"]
    if "test" in task_entry:
        return task_entry["test"]
    raise ValueError("Task entry does not contain train/test loaders.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step-CKA comparison (model_t vs model_{t-1}).")
    parser.add_argument("--dataset", default="cifar100", type=str)
    parser.add_argument("--n_tasks", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--method", default="mose", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--task_ids", default="all", type=str)
    parser.add_argument("--checkpoints_ori", default="cka_data/ori", type=str)
    parser.add_argument("--checkpoints_ours", default="cka_data/ours", type=str)
    parser.add_argument("--output_dir", default="./outputs/cka_step", type=str)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--prefer_test", action="store_true")
    parser.add_argument("--agg", default="deep", choices=["mean", "deep", "layer"], type=str)
    parser.add_argument("--layer_id", default=1, type=int)
    return parser


def _as_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def _compute_layerwise_cka(
    model_factory,
    init_path: str,
    final_path: str,
    loader,
    *,
    device: Optional[torch.device],
    max_batches: Optional[int],
) -> List[float]:
    from models.cka_utils import layerwise_cka

    init_ckpt = torch.load(init_path, map_location="cpu")
    final_ckpt = torch.load(final_path, map_location="cpu")
    init_state = _as_state_dict(init_ckpt)
    final_state = _as_state_dict(final_ckpt)

    model_init = model_factory()
    model_final = model_factory()
    model_init.load_state_dict(init_state, strict=False)
    model_final.load_state_dict(final_state, strict=False)
    model_init.eval()
    model_final.eval()

    if device is not None:
        model_init = model_init.to(device)
        model_final = model_final.to(device)

    with torch.no_grad():
        return layerwise_cka(
            model_init,
            model_final,
            loader,
            device=device,
            max_batches=max_batches,
        )


def _aggregate_layerwise_scores(layer_scores: np.ndarray, agg: str, layer_id: int) -> float:
    arr = np.asarray(layer_scores, dtype=float)
    if arr.size == 0:
        return float("nan")
    if agg == "mean":
        return float(np.nanmean(arr))
    if agg == "deep":
        L = arr.shape[0]
        start = max(0, int(np.floor(2 * L / 3)))
        return float(np.nanmean(arr[start:]))
    if agg == "layer":
        idx = int(layer_id) - 1
        idx = max(0, min(idx, arr.shape[0] - 1))
        return float(arr[idx])
    raise ValueError(f"Unknown agg: {agg}")


def _save_matrix_csv(path: str, data: np.ndarray, task_ids: List[int]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task"] + [f"layer_{i + 1}" for i in range(data.shape[1])]
        writer.writerow(header)
        for row_idx, task_id in enumerate(task_ids):
            writer.writerow([task_id] + data[row_idx].tolist())


def _save_line_plot(
    task_ids: List[int],
    baseline_y: List[float],
    ours_y: List[float],
    save_path: str,
    *,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.0,
            "legend.fontsize": 8,
        }
    )

    x = np.array(task_ids, dtype=int)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=250)
    ax.plot(x, baseline_y, label="baseline", linewidth=1.6)
    ax.plot(x, ours_y, label="ours", linewidth=1.6)
    ax.set_xlabel("Task number")
    ax.set_ylabel(ylabel)
    if len(task_ids) <= 15:
        ax.set_xticks(x)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()

    png_path = save_path if save_path.endswith(".png") else f"{save_path}.png"
    pdf_path = png_path.replace(".png", ".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def _save_delta_line_plot(
    task_ids: List[int],
    delta_y: List[float],
    save_path: str,
    *,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.0,
            "legend.fontsize": 8,
        }
    )

    x = np.array(task_ids, dtype=int)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=250)
    ax.plot(x, delta_y, label="ours - baseline", linewidth=1.6, color="#1b9e77")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Task number")
    ax.set_ylabel(ylabel)
    if len(task_ids) <= 15:
        ax.set_xticks(x)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()

    png_path = save_path if save_path.endswith(".png") else f"{save_path}.png"
    pdf_path = png_path.replace(".png", ".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    repo_root = _find_repo_root(project_dir)
    if repo_root not in sys.path:
        sys.path.insert(1, repo_root)

    from experiment.dataset import get_data
    from models import get_model

    args = _build_arg_parser().parse_args()

    max_batches: Optional[int] = args.max_batches
    if max_batches is not None and max_batches < 0:
        max_batches = None

    device = torch.device("cpu")
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda")

    _, class_num, _, task_loader, _ = get_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        n_tasks=args.n_tasks,
    )

    task_ids = _parse_task_ids(args.task_ids, args.n_tasks)
    task_ids = [t for t in task_ids if t > 0]
    out_root = os.path.join(os.path.abspath(args.output_dir), f"seed_{args.seed}")
    os.makedirs(out_root, exist_ok=True)

    model_factory = lambda: get_model(method_name=args.method, nclasses=class_num)

    ours_scores_all: List[List[float]] = []
    base_scores_all: List[List[float]] = []

    for task_id in task_ids:
        # Step-CKA uses task t-1 data as the probe.
        loader = _pick_task_loader(task_loader[task_id - 1], args.prefer_test)

        ours_prev = _resolve_ckpt_path(args.checkpoints_ours, args.seed, task_id - 1)
        ours_cur = _resolve_ckpt_path(args.checkpoints_ours, args.seed, task_id)
        base_prev = _resolve_ckpt_path(args.checkpoints_ori, args.seed, task_id - 1)
        base_cur = _resolve_ckpt_path(args.checkpoints_ori, args.seed, task_id)

        ours_scores = _compute_layerwise_cka(
            model_factory,
            ours_prev,
            ours_cur,
            loader,
            device=device,
            max_batches=max_batches,
        )
        base_scores = _compute_layerwise_cka(
            model_factory,
            base_prev,
            base_cur,
            loader,
            device=device,
            max_batches=max_batches,
        )

        ours_scores_all.append(ours_scores)
        base_scores_all.append(base_scores)

        print(
            f"[StepCKA] task {task_id}: ours({os.path.basename(ours_prev)}->"
            f"{os.path.basename(ours_cur)}), baseline({os.path.basename(base_prev)}->"
            f"{os.path.basename(base_cur)})"
        )

    ours_matrix = np.array(ours_scores_all, dtype=float)  # [T-1, L]
    base_matrix = np.array(base_scores_all, dtype=float)
    delta_matrix = ours_matrix - base_matrix

    _save_matrix_csv(os.path.join(out_root, "cka_step_ours.csv"), ours_matrix, task_ids)
    _save_matrix_csv(os.path.join(out_root, "cka_step_baseline.csv"), base_matrix, task_ids)
    _save_matrix_csv(os.path.join(out_root, "cka_step_delta.csv"), delta_matrix, task_ids)

    ours_curve = [
        _aggregate_layerwise_scores(ours_matrix[i], args.agg, args.layer_id)
        for i in range(ours_matrix.shape[0])
    ]
    base_curve = [
        _aggregate_layerwise_scores(base_matrix[i], args.agg, args.layer_id)
        for i in range(base_matrix.shape[0])
    ]
    delta_curve = [o - b for o, b in zip(ours_curve, base_curve)]

    if args.agg == "mean":
        ylabel = "Step-CKA (mean over layers)"
    elif args.agg == "deep":
        ylabel = "Step-CKA (mean over last 1/3 layers)"
    else:
        ylabel = f"Step-CKA (layer {args.layer_id})"

    _save_line_plot(task_ids, base_curve, ours_curve, os.path.join(out_root, "cka_step_line.png"), ylabel=ylabel)
    _save_delta_line_plot(task_ids, delta_curve, os.path.join(out_root, "cka_step_delta.png"), ylabel="ΔStep-CKA")

    print(f"[StepCKA] outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
