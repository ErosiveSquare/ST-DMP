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


def _resolve_final_path(root: str, seed: int, n_tasks: int) -> str:
    tried = []
    for seed_dir in _seed_dir_candidates(root, seed):
        patterns = [
            os.path.join(seed_dir, "ckpt_final.pth"),
            os.path.join(seed_dir, "ckpt_final.pt"),
            os.path.join(seed_dir, "final.pth"),
            os.path.join(seed_dir, "final.pt"),
        ]
        tried.append(seed_dir)
        for path in patterns:
            if os.path.isfile(path):
                return path
    return _resolve_ckpt_path(root, seed, n_tasks - 1)


def _pick_task_loader(task_entry: dict, prefer_test: bool) -> torch.utils.data.DataLoader:
    if prefer_test and "test" in task_entry:
        return task_entry["test"]
    if "train" in task_entry:
        return task_entry["train"]
    if "test" in task_entry:
        return task_entry["test"]
    raise ValueError("Task entry does not contain train/test loaders.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-wise CKA comparison for MOSE checkpoints (single line plot).")
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
    parser.add_argument("--output_dir", default="./outputs/cka_compare", type=str)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--prefer_test", action="store_true")

    # --- 折线图聚合方式（把 layer-wise CKA 压成一个标量）---
    # mean: 所有层平均
    # deep: 只平均最后 1/3 的层（更像看“深层表征是否稳定”）
    # layer: 只画某一层（用 --layer_id 指定，1-indexed）
    parser.add_argument("--agg", default="mean", choices=["mean", "deep", "layer"], type=str)
    parser.add_argument("--layer_id", default=1, type=int, help="Used when --agg=layer (1-indexed).")

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

    # 让结果更稳定（避免 dropout/bn 的训练态影响）
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


def _save_matrix_csv(path: str, data: np.ndarray, task_ids: List[int]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task"] + [f"layer_{i + 1}" for i in range(data.shape[1])]
        writer.writerow(header)
        for row_idx, task_id in enumerate(task_ids):
            writer.writerow([task_id] + data[row_idx].tolist())


def _aggregate_layerwise_scores(layer_scores: np.ndarray, agg: str, layer_id: int) -> float:
    # layer_scores: shape [L]
    arr = np.asarray(layer_scores, dtype=float)
    if arr.size == 0:
        return float("nan")

    if agg == "mean":
        return float(np.nanmean(arr))

    if agg == "deep":
        # 最后 1/3 的层平均（你也可以按需要改成最后 K 层）
        L = arr.shape[0]
        start = max(0, int(np.floor(2 * L / 3)))
        return float(np.nanmean(arr[start:]))

    if agg == "layer":
        idx = int(layer_id) - 1
        idx = max(0, min(idx, arr.shape[0] - 1))
        return float(arr[idx])

    raise ValueError(f"Unknown agg: {agg}")


def _save_task_line_plot(
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

    # 让风格更接近你给的例子：小图、线条清晰、legend 左上
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
    y_base = np.array(baseline_y, dtype=float)
    y_ours = np.array(ours_y, dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=250)  # 类似示例的紧凑比例
    ax.plot(x, y_base, label="baseline", linewidth=1.6)
    ax.plot(x, y_ours, label="ours", linewidth=1.6)

    ax.set_xlabel("Task number")
    ax.set_ylabel(ylabel)

    # 如果 task_ids 连续且不多，直接全显示刻度（和示例一致）
    if len(task_ids) <= 15:
        ax.set_xticks(x)

    ax.legend(loc="upper left", frameon=False)

    # 你是 CKA，通常在 [0,1]；如果你更想放大差异，可以改成不设 ylim
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
    y = np.array(delta_y, dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=250)
    ax.plot(x, y, label="ours - baseline", linewidth=1.6, color="#1b9e77")
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
    out_root = os.path.join(os.path.abspath(args.output_dir), f"seed_{args.seed}")
    os.makedirs(out_root, exist_ok=True)

    ours_final = _resolve_final_path(args.checkpoints_ours, args.seed, args.n_tasks)
    ori_final = _resolve_final_path(args.checkpoints_ori, args.seed, args.n_tasks)

    ours_scores_all: List[List[float]] = []
    baseline_scores_all: List[List[float]] = []

    model_factory = lambda: get_model(method_name=args.method, nclasses=class_num)

    for task_id in task_ids:
        ours_task = _resolve_ckpt_path(args.checkpoints_ours, args.seed, task_id)
        ori_task = _resolve_ckpt_path(args.checkpoints_ori, args.seed, task_id)
        loader = _pick_task_loader(task_loader[task_id], args.prefer_test)

        ours_scores = _compute_layerwise_cka(
            model_factory,
            ours_task,
            ours_final,
            loader,
            device=device,
            max_batches=max_batches,
        )
        base_scores = _compute_layerwise_cka(
            model_factory,
            ori_task,
            ori_final,
            loader,
            device=device,
            max_batches=max_batches,
        )

        ours_scores_all.append(ours_scores)
        baseline_scores_all.append(base_scores)

        print(
            f"[CKA] task {task_id}: ours({os.path.basename(ours_task)}) vs final({os.path.basename(ours_final)}), "
            f"baseline({os.path.basename(ori_task)}) vs final({os.path.basename(ori_final)})"
        )

    ours_matrix = np.array(ours_scores_all, dtype=float)   # [T, L]
    base_matrix = np.array(baseline_scores_all, dtype=float)
    delta_matrix = ours_matrix - base_matrix

    # Only keep delta CKA matrix CSV to avoid extra plots.
    _save_matrix_csv(os.path.join(out_root, "cka_matrix_delta.csv"), delta_matrix, task_ids)

    # Only output delta CKA: shallow (first 3 layers avg) and deep (last layer).
    if delta_matrix.ndim == 2 and delta_matrix.shape[1] > 0:
        shallow_k = min(3, delta_matrix.shape[1])
        shallow_delta = np.nanmean(delta_matrix[:, :shallow_k], axis=1).tolist()
        deep_delta = delta_matrix[:, -1].tolist()

        _save_delta_line_plot(
            task_ids,
            shallow_delta,
            os.path.join(out_root, "cka_delta_shallow.png"),
            ylabel=f"Delta CKA (layers 1-{shallow_k})",
        )
        _save_delta_line_plot(
            task_ids,
            deep_delta,
            os.path.join(out_root, "cka_delta_deep.png"),
            ylabel="Delta CKA (last layer)",
        )

    print(f"[CKA] delta plots saved to: {out_root}")


if __name__ == "__main__":
    main()
