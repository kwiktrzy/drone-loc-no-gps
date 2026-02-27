import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import pickle
import hashlib
from feature_transformers.sift_nn import SitfFT
from feature_transformers.xfeat_nn import XFeatLightGlueFT
from vpr_model import VPRModel
from dataloaders.MapsDataloader import MapsDataModule
import pytorch_lightning as pl
import torch
import faiss
from pathlib import Path
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models import utils
import random
from torch.utils.data import DataLoader
import os
import pandas as pd


def get_device(force_cpu: bool = False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):
    model = VPRModel.load_from_checkpoint(checkpoint_path, strict=False)
    return model


def load_datamodule(VAL_CSV):
    datamodule = MapsDataModule(
        tiles_csv_file_paths=[],
        batch_size=32,
        val_set_names=[VAL_CSV],
    )
    return datamodule


@torch.no_grad()
def extract_descriptors(model, dataset, batch_size=64, num_workers=4, device="cuda"):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model.eval().to(device)
    all_desc = []

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=(device == "cuda"))

        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                desc, _ = model(imgs)
        else:
            desc, _ = model(imgs)

        all_desc.append(desc.detach().float().cpu())

    return torch.cat(all_desc, dim=0)


def recall_at_k_from_predictions(predictions, positives, k_values=(1, 5, 10)):
    correct_at_k = {k: 0 for k in k_values}
    num_q = len(predictions)

    for q_idx in range(num_q):
        gt = positives[q_idx]
        if len(gt) == 0:
            continue
        for k in k_values:
            if np.any(np.in1d(predictions[q_idx, :k], gt)):
                correct_at_k[k] += 1

    out = {k: 100.0 * correct_at_k[k] / max(1, num_q) for k in k_values}
    return out


def recall_at_1_by_distance_utm(val_dataset, retrieved_top1, threshold_meters=25.0):
    q_xy = val_dataset.q_utm_np  # [num_q, 2]
    db_xy = val_dataset.db_utm_np  # [num_db, 2]

    dists = np.linalg.norm(q_xy - db_xy[retrieved_top1], axis=1)
    return float((dists < threshold_meters).mean() * 100.0), dists


def calculate_recall(predictions, gt, k_values, dataset_name="dataset"):
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    table = PrettyTable()
    table.field_names = ["K"] + [str(k) for k in k_values]
    table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in correct_at_k])
    print(table.get_string(title=f"Performances on {dataset_name}"))

    return d


def plot_rerank_result(
    query_path,
    candidate_paths,
    sift_scores,
    out_path,
    title=None,
    baseline_top1_path=None,
    gt_paths_set=None,
    query_place_id=None,
    candidate_place_ids=None,
    gt_place_id=None,
):
    from PIL import Image, ImageDraw
    import matplotlib.patches as mpatches

    def add_border(img, color, width=8):
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(width):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
        return img

    K = len(candidate_paths)
    cols = min(6, K + 1)
    rows = int(np.ceil((K + 1) / cols))

    fig = plt.figure(figsize=(3.5 * cols, 4.5 * rows))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── QUERY ──
    ax = fig.add_subplot(rows, cols, 1)
    q_img = Image.open(query_path).convert("RGB")
    add_border(q_img, "blue", width=6)
    ax.imshow(q_img)
    q_title = "QUERY"
    if query_place_id is not None:
        q_title += f"\npid={query_place_id}"
    ax.set_title(q_title, fontsize=9, fontweight="bold", color="blue")
    ax.axis("off")

    # ── RERANKED CANDIDATES ──
    for i, (p, s) in enumerate(zip(candidate_paths, sift_scores), start=2):
        ax = fig.add_subplot(rows, cols, i)
        c_img = Image.open(p).convert("RGB")

        labels = []
        border_color = "gray"
        title_color = "black"

        is_gt = gt_paths_set is not None and p in gt_paths_set
        is_baseline = baseline_top1_path is not None and p == baseline_top1_path
        is_rerank_top1 = i == 2

        if is_gt:
            border_color = "green"
            title_color = "green"
            labels.append("[GT]")
        if is_baseline:
            labels.append("[BASE#1]")
            if not is_gt:
                border_color = "orange"
                title_color = "darkorange"
        if is_rerank_top1:
            labels.append("[SIFT#1]")
            if not is_gt and not is_baseline:
                border_color = "dodgerblue"
                title_color = "dodgerblue"

        add_border(c_img, border_color, width=6)
        ax.imshow(c_img)

        tag = " ".join(labels) if labels else ""
        pid_str = ""
        if candidate_place_ids is not None and (i - 2) < len(candidate_place_ids):
            pid_str = f"\npid={candidate_place_ids[i - 2]}"

        ax.set_title(
            f"#{i-1} s={s:.0f} L: {tag}{pid_str}", fontsize=7, color=title_color
        )
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(facecolor="green", edgecolor="black", label="Ground Truth (GT)"),
        mpatches.Patch(facecolor="orange", edgecolor="black", label="Baseline #1"),
        mpatches.Patch(facecolor="dodgerblue", edgecolor="black", label="SIFT #1"),
        mpatches.Patch(facecolor="gray", edgecolor="black", label="Other candidate"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.9
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_distance_errors(val_dataset, baseline_top1, reranked_top1, top_n_worst=3):
    q_xy = val_dataset.q_utm_np
    db_xy = val_dataset.db_utm_np

    dist_baseline = np.linalg.norm(q_xy - db_xy[baseline_top1], axis=1)
    dist_reranked = np.linalg.norm(q_xy - db_xy[reranked_top1], axis=1)

    dist_diff = dist_reranked - dist_baseline

    worst_indices = np.argsort(dist_diff)[::-1][:top_n_worst]

    return {
        "avg_dist_baseline": float(np.mean(dist_baseline)),
        "avg_dist_reranked": float(np.mean(dist_reranked)),
        "dist_diff": dist_diff,
        "dist_baseline": dist_baseline,
        "dist_reranked": dist_reranked,
        "worst_indices": worst_indices,
    }


def rerank_topk(
    val_dataset,
    predictions,
    positives,
    matcher_instance,
    match_method_name: str,
    method_label: str,
    topk=10,
    visualize=False,
    out_dir=None,
    max_vis=20,
    seed=0,
    path_to_pid=None,
):
    match_func = getattr(matcher_instance, match_method_name)

    num_q = predictions.shape[0]
    topk = min(topk, predictions.shape[1])

    reranked = np.zeros((num_q, topk), dtype=np.int64)
    best_top1 = np.zeros((num_q,), dtype=np.int64)
    scores_debug = []

    vis_q_idx = set()
    if visualize:
        random.seed(seed)
        vis_q_idx = set(random.sample(range(num_q), k=min(max_vis, num_q)))
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

    for q_idx in tqdm(range(num_q), desc=f"Reranking with {method_label}"):
        q_path = val_dataset.q_image_paths[q_idx]
        cand_db_idx = predictions[q_idx, :topk].tolist()

        scores = []
        for db_i in cand_db_idx:
            c_path = val_dataset.db_image_paths[db_i]
            score = match_func(
                img_query_path=q_path,
                img_candidate_path=c_path,
                visualize=False,
            )
            scores.append(float(score))

        order = np.argsort(scores)[::-1]
        reranked[q_idx] = np.array([cand_db_idx[i] for i in order], dtype=np.int64)
        best_top1[q_idx] = reranked[q_idx, 0]
        scores_debug.append([scores[i] for i in order])

        if visualize and (q_idx in vis_q_idx) and out_dir is not None:
            gt_set = set(val_dataset.db_image_paths[i] for i in positives[q_idx])
            baseline_top1_path = val_dataset.db_image_paths[cand_db_idx[0]]

            q_pid = None
            cand_pids = []
            gt_pid = None

            if path_to_pid is not None:
                q_pid = path_to_pid.get(q_path)
                cand_paths = [val_dataset.db_image_paths[i] for i in reranked[q_idx]]
                cand_pids = [path_to_pid.get(p) for p in cand_paths]

                if len(positives[q_idx]) > 0:
                    gt_path_first = val_dataset.db_image_paths[positives[q_idx][0]]
                    gt_pid = path_to_pid.get(gt_path_first)

            plot_rerank_result(
                query_path=q_path,
                candidate_paths=[
                    val_dataset.db_image_paths[i] for i in reranked[q_idx]
                ],
                sift_scores=scores_debug[-1],
                out_path=os.path.join(out_dir, f"q_{q_idx:05d}.png"),
                title=f"Q{q_idx} — rerank top{topk} ({method_label})"
                + (f" | GT pid={gt_pid}" if gt_pid is not None else ""),
                baseline_top1_path=baseline_top1_path,
                gt_paths_set=gt_set,
                query_place_id=q_pid,
                candidate_place_ids=cand_pids,
                gt_place_id=gt_pid,
            )

    return reranked, best_top1, scores_debug


# ─────────────────────────────────────────────────────────
#  CACHE HELPERS
# ─────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(current_dir, ".eval_cache")


def _file_hash(path: str, algo="sha256", chunk_size=1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(ckpt_path: str, val_csv: str) -> str:
    ckpt_hash = _file_hash(ckpt_path)
    ds_hash = _file_hash(val_csv)
    return f"{ckpt_hash}_{ds_hash}"


def save_cache(key: str, tag: str, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [cache] saved  → {path}")


def load_cache(key: str, tag: str):
    path = os.path.join(CACHE_DIR, f"{key}_{tag}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  [cache] loaded ← {path}")
    return data


def main():
    CHECKPOINT_PATH = (
        "/home/user/PycharmProjects/mgr-repack/ckpts/resnet50_v2_(39)_R1[0.3675].ckpt"
    )
    VAL_CSV = "/home/user/PycharmProjects/mgr-repack/Dataframes/one_to_one/Changjiang-23-v1_one_to_one.csv"
    df = pd.read_csv(VAL_CSV)
    path_to_pid = dict(zip(df["img_path"], df["place_id"]))
    force_cpu = True
    device = get_device(force_cpu=force_cpu)

    model = load_model(CHECKPOINT_PATH)
    dm = load_datamodule(VAL_CSV)
    dm.setup(stage="validate")

    val_dataset = dm.val_datasets[0]
    short_val_name = Path(dm.val_set_names[0]).stem

    # ── key cache ──────────────────────────────────────
    cache_key = _cache_key(CHECKPOINT_PATH, VAL_CSV)
    print(f"Cache key: {cache_key}")

    # ── trainer.validate  (cache) ────────────────────
    val_results = load_cache(cache_key, "trainer_validate")
    if val_results is None:
        if device == "cuda":
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                precision="16-mixed",
                logger=False,
            )
        else:
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                precision=32,
                logger=False,
            )
        val_results = trainer.validate(model=model, datamodule=dm)
        save_cache(cache_key, "trainer_validate", val_results)
    else:
        print("  ↳ trainer.validate() skipped – using cache")
    print("Trainer validate results:", val_results)

    feats = load_cache(cache_key, "descriptors")
    if feats is None:
        feats = extract_descriptors(
            model=model,
            dataset=val_dataset,
            batch_size=64,
            num_workers=0 if device == "cpu" else 4,
            device=device,
        )
        save_cache(cache_key, "descriptors", feats)
    else:
        print("  ↳ extract_descriptors() skipped – using cache")

    num_references = len(val_dataset.db_image_paths)
    r_list = feats[:num_references].contiguous()
    q_list = feats[num_references:].contiguous()

    positives = val_dataset.get_positives()
    print(f"\nDataset: {short_val_name}")
    print(f"References: {num_references}, Queries: {len(val_dataset.q_image_paths)}")

    # ── baseline FAISS  (cache) ──────────────────────
    baseline_predictions = load_cache(cache_key, "baseline_predictions")
    if baseline_predictions is None:
        K = 10
        baseline_predictions = utils.get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10],
            gt=positives,
            print_results=False,
            dataset_name=short_val_name,
            faiss_gpu=(device == "cuda") and getattr(model, "faiss_gpu", False),
            testing=True,
        )
        save_cache(cache_key, "baseline_predictions", baseline_predictions)
    else:
        print("  ↳ baseline FAISS skipped – using cache")

    baseline_recalls = recall_at_k_from_predictions(
        baseline_predictions, positives, k_values=(1, 5, 10)
    )
    baseline_top1 = np.asarray(baseline_predictions[:, 0])
    baseline_r1_25m, _ = recall_at_1_by_distance_utm(
        val_dataset, baseline_top1, threshold_meters=20.0
    )

    K = 10

    sift_matcher = SitfFT()
    xfeat_matcher = XFeatLightGlueFT()

    rerank_methods = {
        "SIFT": (sift_matcher, "get_sift_match_score"),
        "XFEAT_BF": (xfeat_matcher, "get_xfeat_bf_match_score"),
        "XFEAT_LG": (xfeat_matcher, "get_xfeat_lightglue_match_score"),
    }

    # ── eval loop ─────────────────────────

    for method_label, (matcher_inst, match_method_name) in rerank_methods.items():
        print(f"\n{'-'*60}")
        print(f" Start evaluation for: {method_label}")
        print(f"{'-'*60}")

        cache_tag = f"rerank_{method_label.lower()}"
        out_dir = f"debug_{method_label.lower()}_{short_val_name}"
        method_cache = load_cache(cache_key, cache_tag)

        if method_cache is None:
            reranked_preds, best_top1, scores_debug = rerank_topk(
                val_dataset=val_dataset,
                predictions=baseline_predictions,
                positives=positives,
                matcher_instance=matcher_inst,
                match_method_name=match_method_name,
                method_label=method_label,
                topk=K,
                visualize=True,
                out_dir=out_dir,
                max_vis=20,
                seed=0,
                path_to_pid=path_to_pid,
            )
            save_cache(
                cache_key,
                cache_tag,
                {
                    "reranked": reranked_preds,
                    "top1": best_top1,
                    "scores": scores_debug,
                },
            )
        else:
            print(f"  ↳ {method_label} rerank skipped – using cache")
            reranked_preds = method_cache["reranked"]
            best_top1 = method_cache["top1"]

        best_top1 = np.asarray(best_top1)
        reranked_recalls = recall_at_k_from_predictions(
            reranked_preds, positives, k_values=(1, 5, 10)
        )
        reranked_r1_25m, _ = recall_at_1_by_distance_utm(
            val_dataset, best_top1, threshold_meters=20.0
        )

        num_total = len(best_top1)
        baseline_hit = np.array(
            [int(baseline_top1[i]) in positives[i] for i in range(num_total)]
        )
        reranked_hit = np.array(
            [int(best_top1[i]) in positives[i] for i in range(num_total)]
        )

        baseline_correct = int(baseline_hit.sum())
        reranked_correct = int(reranked_hit.sum())

        improved = int((~baseline_hit & reranked_hit).sum())
        worsened = int((baseline_hit & ~reranked_hit).sum())
        changed = int((baseline_top1 != best_top1).sum())

        # ── raport ──
        print(f"\n  [Result: {method_label}]")
        print(
            f"  BASELINE top-1 accuracy:  {baseline_correct}/{num_total}  ({100*baseline_correct/num_total:.2f}%)"
        )
        print(
            f"  RERANKED top-1 accuracy:  {reranked_correct}/{num_total}  ({100*reranked_correct/num_total:.2f}%)"
        )
        print(f"\n  Top-1 changed:            {changed}/{num_total}")
        print(f"    ✅ improvement (miss→hit): {improved}")
        print(f"    ❌ worsened (hit→miss): {worsened}")
        print(f"    ➡  net gain:              {improved - worsened:+d}")

        print(
            f"\n  BASELINE Recall@K:  {{{', '.join(f'R@{k}: {v:.2f}%' for k,v in baseline_recalls.items())}}}"
        )
        print(
            f"  RERANKED Recall@K:  {{{', '.join(f'R@{k}: {v:.2f}%' for k,v in reranked_recalls.items())}}}"
        )

        print(f"\n  BASELINE R@1 (<20m UTM):  {baseline_r1_25m:.2f}%")
        print(f"  RERANKED R@1 (<20m UTM):  {reranked_r1_25m:.2f}%")

        # Distance analysis
        dist_analysis = analyze_distance_errors(
            val_dataset=val_dataset,
            baseline_top1=baseline_top1,
            reranked_top1=best_top1,
            top_n_worst=3,
        )

        print(f"\n  [Distance analysis: {method_label}]")
        print(
            f"  Mean error distance BASELINE: {dist_analysis['avg_dist_baseline']:.2f} m"
        )
        print(
            f"  Mean error distance RERANKED: {dist_analysis['avg_dist_reranked']:.2f} m"
        )
        diff_avg = (
            dist_analysis["avg_dist_reranked"] - dist_analysis["avg_dist_baseline"]
        )
        print(f"  Mean distance change:       {diff_avg:+.2f} m (less = better)")

        print(f"\n  [TOP 3 WORST INDICES FOR {method_label}]")
        for rank, q_idx in enumerate(dist_analysis["worst_indices"], start=1):
            b_dist = dist_analysis["dist_baseline"][q_idx]
            r_dist = dist_analysis["dist_reranked"][q_idx]
            pogorszenie = dist_analysis["dist_diff"][q_idx]

            if pogorszenie <= 0:
                break

            q_path = val_dataset.q_image_paths[q_idx]
            baseline_path = val_dataset.db_image_paths[baseline_top1[q_idx]]
            reranked_path = val_dataset.db_image_paths[best_top1[q_idx]]

            print(f"    {rank}. Query (q_idx = {q_idx}):")
            print(
                f"       Error Baseline: {b_dist:.1f}m -> Error Reranked: {r_dist:.1f}m (Worse {pogorszenie:.1f}m)"
            )
            print(f"       📸 Query:    {q_path}")
            print(f"       ✅ Baseline: {baseline_path}")
            print(f"       ❌ Reranked: {reranked_path}\n")


# todo: verify batcher for validation!


if __name__ == "__main__":
    main()
