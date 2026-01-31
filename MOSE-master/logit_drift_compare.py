import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
    parser = argparse.ArgumentParser(description="Logit drift and prediction flip rate comparison.")
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
    parser.add_argument("--output_dir", default="./outputs/logit_drift", type=str)
    parser.add_argument("--max_batches", default=-1, type=int, help="Max batches per probe loader.")
    parser.add_argument("--prefer_test", action="store_true")
    parser.add_argument("--head", default="mean", choices=["mean", "expert"], type=str)
    parser.add_argument("--expert_id", default=3, type=int)
    parser.add_argument("--probe_mode", default="prev", choices=["prev", "all_old"], type=str)
    return parser


def _as_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def _load_model(model_factory, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _as_state_dict(ckpt)
    model = model_factory()
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def _extract_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    head: str,
    expert_id: int,
) -> torch.Tensor:
    out = model(x)
    logits_list = out
    if isinstance(out, tuple):
        for candidate in reversed(out):
            if isinstance(candidate, (list, tuple)):
                logits_list = candidate
                break
    if isinstance(logits_list, torch.Tensor):
        return logits_list
    if head == "mean":
        return torch.stack(list(logits_list), dim=1).mean(dim=1)
    idx = max(0, min(expert_id, len(logits_list) - 1))
    return logits_list[idx]


def _iter_probe_batches(
    task_loader: list,
    task_id: int,
    *,
    prefer_test: bool,
    probe_mode: str,
    max_batches: Optional[int],
):
    if probe_mode == "prev":
        probe_tasks = [task_id - 1]
    else:
        probe_tasks = list(range(task_id))
    for t in probe_tasks:
        loader = _pick_task_loader(task_loader[t], prefer_test)
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            yield batch


def _accumulate_metrics(
    model_t: torch.nn.Module,
    model_prev: torch.nn.Module,
    task_loader: list,
    task_id: int,
    *,
    device: torch.device,
    head: str,
    expert_id: int,
    prefer_test: bool,
    probe_mode: str,
    max_batches: Optional[int],
) -> Tuple[float, float, float]:
    sum_kl = 0.0
    sum_l2 = 0.0
    sum_flip = 0.0
    count = 0

    for batch in _iter_probe_batches(
        task_loader, task_id, prefer_test=prefer_test, probe_mode=probe_mode, max_batches=max_batches
    ):
        x = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            logits_t = _extract_logits(model_t, x, head=head, expert_id=expert_id)
            logits_prev = _extract_logits(model_prev, x, head=head, expert_id=expert_id)
            prob_prev = F.softmax(logits_prev, dim=1)
            kl = F.kl_div(F.log_softmax(logits_t, dim=1), prob_prev, reduction="batchmean")
            l2 = torch.norm(logits_t - logits_prev, dim=1).mean()
            flip = (logits_t.argmax(dim=1) != logits_prev.argmax(dim=1)).float().mean()

        bs = x.size(0)
        sum_kl += float(kl.item()) * bs
        sum_l2 += float(l2.item()) * bs
        sum_flip += float(flip.item()) * bs
        count += bs

    if count == 0:
        return float("nan"), float("nan"), float("nan")
    return sum_kl / count, sum_l2 / count, sum_flip / count


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

    ours_kl, ours_l2, ours_flip = [], [], []
    base_kl, base_l2, base_flip = [], [], []

    for task_id in task_ids:
        ours_prev = _resolve_ckpt_path(args.checkpoints_ours, args.seed, task_id - 1)
        ours_cur = _resolve_ckpt_path(args.checkpoints_ours, args.seed, task_id)
        base_prev = _resolve_ckpt_path(args.checkpoints_ori, args.seed, task_id - 1)
        base_cur = _resolve_ckpt_path(args.checkpoints_ori, args.seed, task_id)

        ours_model_prev = _load_model(model_factory, ours_prev, device)
        ours_model_cur = _load_model(model_factory, ours_cur, device)
        base_model_prev = _load_model(model_factory, base_prev, device)
        base_model_cur = _load_model(model_factory, base_cur, device)

        ours_metrics = _accumulate_metrics(
            ours_model_cur,
            ours_model_prev,
            task_loader,
            task_id,
            device=device,
            head=args.head,
            expert_id=args.expert_id,
            prefer_test=args.prefer_test,
            probe_mode=args.probe_mode,
            max_batches=max_batches,
        )
        base_metrics = _accumulate_metrics(
            base_model_cur,
            base_model_prev,
            task_loader,
            task_id,
            device=device,
            head=args.head,
            expert_id=args.expert_id,
            prefer_test=args.prefer_test,
            probe_mode=args.probe_mode,
            max_batches=max_batches,
        )

        ours_kl.append(ours_metrics[0])
        ours_l2.append(ours_metrics[1])
        ours_flip.append(ours_metrics[2])
        base_kl.append(base_metrics[0])
        base_l2.append(base_metrics[1])
        base_flip.append(base_metrics[2])

        print(
            f"[LogitDrift] task {task_id}: ours(kl={ours_metrics[0]:.4f}, l2={ours_metrics[1]:.4f}, flip={ours_metrics[2]:.4f}) "
            f"baseline(kl={base_metrics[0]:.4f}, l2={base_metrics[1]:.4f}, flip={base_metrics[2]:.4f})"
        )

    # CSV
    csv_path = os.path.join(out_root, "logit_drift.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(["task", "ours_kl", "base_kl", "ours_l2", "base_l2", "ours_flip", "base_flip"])
        for idx, task_id in enumerate(task_ids):
            writer.writerow([task_id, ours_kl[idx], base_kl[idx], ours_l2[idx], base_l2[idx], ours_flip[idx], base_flip[idx]])

    _save_line_plot(task_ids, base_kl, ours_kl, os.path.join(out_root, "logit_kl_drift.png"), ylabel="KL(p_t || p_{t-1})")
    _save_line_plot(task_ids, base_l2, ours_l2, os.path.join(out_root, "logit_l2_drift.png"), ylabel="L2(z_t - z_{t-1})")
    _save_line_plot(task_ids, base_flip, ours_flip, os.path.join(out_root, "pred_flip_rate.png"), ylabel="Prediction flip rate")

    print(f"[LogitDrift] outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
