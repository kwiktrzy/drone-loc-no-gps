from typing import List, Tuple, Dict
import gc
import json
import os
import shutil
import inspect
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.place_id_generators.ManyToManyPlaceIdGenerator import (
    ManyToManyPlaceIdGenerator,
)
from dataset_splitter.structs.MapSatellite import MapSatellite
from dataset_splitter.satellite_generators.OverlapingTilesGenerator import (
    OverlapingTilesGenerator,
)
from dataset_splitter.uav_generators.UavSmallerCropGenerator import (
    UavSmallerCropGenerator,
)
from vpr_model import VPRModel


class PipelineConfig:
    def __init__(self, project_root="/workspace/"):
        # --- Base Paths ---
        self.PROJECT_ROOT = Path(project_root)
        self.DATASETS_ROOT = self.PROJECT_ROOT / "datasets"
        self.UAV_VISLOC_ROOT = self.DATASETS_ROOT / "UAV_VisLoc_dataset"
        self.AERIAL_VL_ROOT = self.DATASETS_ROOT / "Aerial_VL_dataset"
        self.DATAFRAMES_ROOT = self.PROJECT_ROOT / "drone-loc-no-gps/Dataframes"

        self.DATAFRAMES_ONE_TO_ONE_DIR = self.DATAFRAMES_ROOT / "one_to_one"
        self.DATAFRAMES_OVERLAPPING_PATCHES_DIR = (
            self.DATAFRAMES_ROOT / "overlapping_patches"
        )
        self.DATAFRAMES_TILES_TRASH = self.DATAFRAMES_ROOT / "tiles_trash"
        self.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR = (
            self.DATASETS_ROOT / "train_tiles_one_to_one"
        )
        self.THUMBNAILS_OVERLAPPING_PATCHES_OUTPUT_DIR = (
            self.DATASETS_ROOT / "train_tiles_overlapping_patches"
        )

        # --- Regeneration Flags ---
        self.force_regenerate_tiles = False
        self.force_regenerate_place_ids = False

        # --- Generation Methods ---
        self.one_to_one_tiles = True
        self.overlapping_patches_tiles = False


def clearup_generated_data(
    config: PipelineConfig, output_csv_path: Path, thumb_dir: Path, region_name: str
) -> bool:
    if config.force_regenerate_tiles:
        if output_csv_path.exists():
            print(f"Force regenerate: Removing existing CSV: {output_csv_path}")
            output_csv_path.unlink()
        if thumb_dir.exists():
            print(f"Force regenerate: Removing existing tile directory: {thumb_dir}")
            shutil.rmtree(thumb_dir)
        return False

    if not (
        output_csv_path.exists() and thumb_dir.exists() and any(thumb_dir.iterdir())
    ):
        return False
    try:
        df = pd.read_csv(output_csv_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return False

    col = None
    if "friendly-name" in df.columns:
        col = "friendly-name"
    elif "friendly_name" in df.columns:
        col = "friendly_name"

    if col is None:
        return False

    s = df[col].astype(str)
    has_uav = s.str.contains("-uav").any()
    has_sat = s.str.contains("-satellite").any()

    if has_uav and has_sat:
        print(f"\nSkipping tile generation for '{region_name}', already processed.")
        return True

    return False


def get_processed_path(base_path: str, suffix: str) -> str:
    path_obj = Path(base_path)
    new_filename = f"{path_obj.stem}-{suffix}{path_obj.suffix}"
    return str(path_obj.parent / new_filename)


def build_callbacks(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_mean = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_mean_R1_4sets",
        filename="best_mean-{epoch:02d}-{val_mean_R1_4sets:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
        mode="max",
    )

    checkpoint_shandan_v2 = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="Shandan-v2_one_to_one/R1",
        filename="best_shandan_v2-{epoch:02d}-{Shandan-v2_one_to_one/R1:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode="max",
    )

    checkpoint_changjiang_v2 = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="Changjiang-23-v2_one_to_one/R1",
        filename="best_changjiang_v2-{epoch:02d}-{Changjiang-23-v2_one_to_one/R1:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode="max",
    )

    callbacks = [checkpoint_mean, checkpoint_shandan_v2, checkpoint_changjiang_v2]
    cb_map = {
        "mean": checkpoint_mean,
        "shandan_v2": checkpoint_shandan_v2,
        "changjiang_v2": checkpoint_changjiang_v2,
    }
    return callbacks, cb_map


def score_to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def create_detailed_attention_figure(
    query_path: str,
    model_data: Dict[str, Dict[str, str]],  # model_name -> {"attn_query": path, "top1": path, "attn_top1": path}
    model_hits: Dict[str, bool],
    model_names_ordered: List[str],
    output_path: Path,
    title: str = "",
):

    n_models = len(model_names_ordered)
    n_rows = n_models
    n_cols = 4  # query orig, query attn, top1 orig, top1 attn

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    query_img = None
    if os.path.exists(query_path):
        query_img = Image.open(query_path).convert("RGB")

    for row_idx, mname in enumerate(model_names_ordered):
        data = model_data.get(mname, {})
        hit = model_hits.get(mname, False)
        status = "HIT" if hit else "MISS"

        ax_q = axes[row_idx, 0]
        if row_idx == 0 and query_img is not None:
            ax_q.imshow(query_img)
            ax_q.set_title("Query (UAV)", fontsize=9, fontweight="bold")
        else:
            ax_q.axis("off")
        ax_q.axis("off")

        ax_q_att = axes[row_idx, 1]
        attn_q_path = data.get("attn_query")
        if attn_q_path and os.path.exists(attn_q_path):
            img = Image.open(attn_q_path).convert("RGB")
            ax_q_att.imshow(img)
            ax_q_att.set_title(f"{mname} Query Attn", fontsize=9)
        else:
            ax_q_att.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax_q_att.axis("off")

        ax_t1 = axes[row_idx, 2]
        top1_path = data.get("top1")
        if top1_path and os.path.exists(top1_path):
            img_t1 = Image.open(top1_path).convert("RGB")
            ax_t1.imshow(img_t1)
        else:
            ax_t1.text(0.5, 0.5, "N/A", ha="center", va="center")
        color = "green" if hit else "red"
        ax_t1.set_title(f"Top-1 ({status})", fontsize=9, color=color, fontweight="bold")
        ax_t1.axis("off")

        ax_t1_att = axes[row_idx, 3]
        attn_t1_path = data.get("attn_top1")
        if attn_t1_path and os.path.exists(attn_t1_path):
            img = Image.open(attn_t1_path).convert("RGB")
            ax_t1_att.imshow(img)
            ax_t1_att.set_title(f"{mname} Top-1 Attn", fontsize=9)
        else:
            ax_t1_att.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax_t1_att.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

# ====================================================================
#  PLACE ID GENERATION
# ====================================================================


def generate_place_ids_for_variant(
    config: PipelineConfig,
    data_config: List[dict],
    use_informativeness_filter: bool = True,
    uav_overlap_multiplier: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """
    Generates (or loads) CSV files with the place_id for a given variant.
    Returns lists of train/val paths.
    """
    train_csvs: List[str] = []
    val_csvs: List[str] = []

    print(
        f"\n=== Generating Place IDs (filter={use_informativeness_filter}, "
        f"overlap_mult={uav_overlap_multiplier}) ==="
    )

    for d_conf in data_config:
        region_name = d_conf["region_name"]
        base_path = str(config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv")
        final_path = get_processed_path(base_path, d_conf["output_suffix"])

        is_val = d_conf.get("set_type") == "val"
        need_generate = (
            config.force_regenerate_place_ids or not Path(final_path).exists()
        )

        if need_generate:
            print(f"  [GENERATING] {region_name} -> {Path(final_path).name}")
            generator = ManyToManyPlaceIdGenerator(
                csv_tiles_path=base_path,
                csv_place_ids_output_path=final_path,
                force_regenerate=True,
                is_validation_set=is_val,
                is_validation_set_v2=d_conf.get("val_variant") == "v2",
                radius_neighbors_meters=70 if is_val else d_conf["crop_range_meters"],
                tiles_trash_directory=config.DATAFRAMES_TILES_TRASH,
                use_informativeness_filter=use_informativeness_filter,
                uav_overlap_multiplier=uav_overlap_multiplier,
            )
            generator.generate_place_ids()
        else:
            print(f"  [EXISTS]    {region_name} -> {Path(final_path).name}")

        if d_conf["set_type"] == "train":
            train_csvs.append(final_path)
        elif d_conf["set_type"] == "val":
            val_csvs.append(final_path)

    return train_csvs, val_csvs


# ====================================================================
#  TRAINING
# ====================================================================


def run_single_experiment(
    exp: dict, train_csvs: List[str], val_csvs: List[str], logs_root: Path
) -> dict:

    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {exp['name']}")
    print(f"  Seed: {exp['seed']}")
    print(f"  LR: {exp.get('lr', 0.03)}")
    print(f"  Agg: {exp.get('agg_arch')}")
    print("=" * 100)

    pl.seed_everything(exp["seed"], workers=True)

    run_dir = (logs_root / exp["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = run_dir / "val_metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    with open(run_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2)

    datamodule = MapsDataModule(
        tiles_csv_file_paths=train_csvs,
        batch_size=exp.get("batch_size", 32),
        val_set_names=val_csvs,
        shuffle_all=True,
    )

    valid_model_args = inspect.signature(VPRModel.__init__).parameters.keys()
    model_kwargs = {k: v for k, v in exp.items() if k in valid_model_args}
    model = VPRModel(**model_kwargs)

    model.save_val_predictions = False
    model.run_dir = str(run_dir)

    callbacks, cb_map = build_callbacks(run_dir)

    old_cwd = os.getcwd()
    os.chdir(run_dir)

    try:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            default_root_dir=".",
            num_nodes=1,
            num_sanity_val_steps=0,
            precision="32",
            max_epochs=exp["max_epochs"],
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,
            log_every_n_steps=10,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0,
        )

        trainer.fit(model=model, datamodule=datamodule)

        best_mean_path = cb_map["mean"].best_model_path
        best_mean_score = score_to_float(cb_map["mean"].best_model_score)

        result = {
            "experiment": exp["name"],
            "seed": exp["seed"],
            "lr": exp.get("lr", 0.03),
            "agg_arch": exp.get("agg_arch"),
            "best_mean_score": best_mean_score,
            "best_mean_path": str(best_mean_path) if best_mean_path else None,
            "run_dir": str(run_dir),
            "max_epochs": exp["max_epochs"],
        }

        if metrics_jsonl.exists():
            with open(metrics_jsonl, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            if lines:
                last = lines[-1]
                result["best_epoch"] = last.get("epoch")
                result["datasets"] = last.get("datasets", {})

        return result

    except Exception as e:

        if "CUDA" in str(e):
            print(f"[CUDA ERROR] Attempting cleanup...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"\n[ERROR] Experiment {exp['name']} failed: {e}")
        return {
            "experiment": exp["name"],
            "seed": exp["seed"],
            "error": str(e),
            "run_dir": str(run_dir),
        }

    finally:
        os.chdir(old_cwd)
        try:
            del trainer
            del model
            del datamodule
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        import time; time.sleep(1)


# ====================================================================
#  COMPARISON REPORT
# ====================================================================


def load_val_datasets_once(val_csvs: List[str]) -> Dict[str, object]:
    dm = MapsDataModule(
        tiles_csv_file_paths=[],
        batch_size=1,
        val_set_names=val_csvs,
        shuffle_all=False,
    )
    dm.setup("validate")

    ds_map = {}
    for path, ds in zip(dm.val_set_names, dm.val_datasets):
        short_name = Path(path).stem
        ds_map[short_name] = ds
    return ds_map


def load_experiment_predictions(
    experiments: List[dict], logs_root: Path
) -> Dict[tuple, dict]:
    """
    key: (exp_name, dataset_short_name)
    """
    all_preds = {}
    for exp in experiments:
        run_dir = logs_root / exp["name"]
        if not run_dir.exists():
            continue
        for pred_file in run_dir.glob("predictions_*.json"):
            ds_name = pred_file.stem.replace("predictions_", "")
            key = (exp["name"], ds_name)
            with open(pred_file, "r", encoding="utf-8") as f:
                all_preds[key] = json.load(f)
    return all_preds


def find_divergent_and_consensus(
    predictions_by_model: Dict[str, dict],
    num_divergent: int = 10,
    num_consensus: int = 5,
) -> Tuple[list, list]:
    """
    For a single validation set, it compares the predictions of all models.
    Returns (divergent_queries, consensus_queries).

    Divergent: The top-1 result differs from at least two other models.
    Consensus: All models hit the top 1 (or all missed).
    """
    model_names = sorted(predictions_by_model.keys())
    if len(model_names) < 2:
        return [], []

    model_qmaps = {}
    for mname in model_names:
        preds = predictions_by_model[mname]
        qmap = {q["query_path"]: q for q in preds.get("queries", [])}
        model_qmaps[mname] = qmap

    common_paths = set(model_qmaps[model_names[0]].keys())
    for mname in model_names[1:]:
        common_paths &= set(model_qmaps[mname].keys())

    divergent = []
    consensus_hits = []

    for qpath in common_paths:
        top1s = {}
        hits = {}
        for mname in model_names:
            q = model_qmaps[mname][qpath]
            top1_path = q.get("top5_paths", [None])[0] if q.get("top5_paths") else None
            top1s[mname] = top1_path
            hits[mname] = q.get("is_hit_r1", False)

        unique_top1s = set(t for t in top1s.values() if t is not None)

        entry = {
            "query_path": qpath,
            "top1s": top1s,
            "hits": hits,
        }

        if len(unique_top1s) >= 2:
            divergent.append(entry)
        elif all(hits.values()) or not any(hits.values()):
            consensus_hits.append(entry)

    divergent.sort(
        key=lambda e: len(set(t for t in e["top1s"].values() if t)), reverse=True
    )

    return divergent[:num_divergent], consensus_hits[:num_consensus]


def create_multimodel_comparison_figure(
    query_path: str,
    model_top1s: Dict[str, str],
    model_hits: Dict[str, str],
    model_names_ordered: list,
    output_path: Path,
    title: str = "",
):
    try:
        n_models = len(model_names_ordered)
        n_cols = 1 + n_models  # query + modeli
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        # Query
        if os.path.exists(query_path):
            img_q = Image.open(query_path).convert("RGB")
            axes[0].imshow(img_q)
        axes[0].set_title("Query (UAV)", fontweight="bold", fontsize=10)
        axes[0].axis("off")

        # Modele
        for i, mname in enumerate(model_names_ordered):
            ax = axes[i + 1]
            top1_path = model_top1s.get(mname)
            hit = model_hits.get(mname, False)
            status = "HIT" if hit else "MISS"

            if top1_path and os.path.exists(top1_path):
                img = Image.open(top1_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)

            color = "green" if hit else "red"
            ax.set_title(
                f"{mname}\n{status}", fontsize=9, color=color, fontweight="bold"
            )
            ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return True

    except Exception as e:
        print(f"[ERROR] create_multimodel_comparison_figure: {e}")
        plt.close("all")
        return False


def create_multimodel_heatmap_figure(
    query_path: str,
    model_attn_paths: Dict[str, str],
    model_names_ordered: list,
    output_path: Path,
    title: str = "",
):
    try:
        n_models = len(model_names_ordered)
        n_cols = 1 + n_models
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        # Query
        if os.path.exists(query_path):
            img_q = Image.open(query_path).convert("RGB")
            axes[0].imshow(img_q)
        axes[0].set_title("Query (UAV)", fontweight="bold", fontsize=10)
        axes[0].axis("off")

        # Heatmapy per model
        for i, mname in enumerate(model_names_ordered):
            ax = axes[i + 1]
            attn_path = model_attn_paths.get(mname)

            if attn_path and os.path.exists(attn_path):
                img = Image.open(attn_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{mname}\nAttention", fontsize=9, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No attention", ha="center", va="center", fontsize=12)
                ax.set_title(f"{mname}\nN/A", fontsize=9)

            ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return True

    except Exception as e:
        print(f"[ERROR] create_multimodel_heatmap_figure: {e}")
        plt.close("all")
        return False


def run_final_validations(
    experiments: List[dict],
    logs_root: Path,
    val_csvs: List[str],
    ds_map: Dict[str, object],
) -> Dict[str, VPRModel]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = {}

    for exp in experiments:
        exp_name = exp["name"]
        run_dir = logs_root / exp_name
        summary_path = run_dir / "summary.json"

        # Wczytaj zapisane metryki z treningu
        if not summary_path.exists():
            print(f"[SKIP] No summary for {exp_name}")
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        best_path = summary.get("best_mean_path")
        if not best_path or not Path(best_path).exists():
            print(f"[SKIP] No checkpoint for {exp_name}")
            continue

        all_preds_exist = True
        for ds_name in ds_map:
            pred_file = run_dir / f"predictions_{ds_name}.json"
            if not pred_file.exists():
                all_preds_exist = False
                break

        if all_preds_exist:
            print(f"[SKIP] Predictions already exist for {exp_name}")
            model = VPRModel.load_from_checkpoint(best_path, strict=True)
            model.to(device)
            model.eval()
            loaded_models[exp_name] = model
            continue

        # Final validation
        print(f"\n[FINAL VAL] {exp_name}")
        model = VPRModel.load_from_checkpoint(best_path, strict=True)
        model.save_val_predictions = True
        model.run_dir = str(run_dir)
        model.is_final_validation = True
        model.to(device)
        model.eval()

        val_dm = MapsDataModule(
            tiles_csv_file_paths=[],
            batch_size=1,
            val_set_names=val_csvs,
            shuffle_all=False,
        )

        val_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            num_sanity_val_steps=0,
            precision="32",
        )
        val_dm.setup("validate")

        val_trainer.validate(model, datamodule=val_dm)
        del val_trainer, val_dm
        torch.cuda.empty_cache()

        loaded_models[exp_name] = model

    return loaded_models


def generate_comparison_report(
    experiments: List[dict],
    logs_root: Path,
    val_csvs: List[str],
    num_divergent: int = 10,
    num_consensus: int = 5,
):

    report_dir = logs_root / "comparison_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[REPORT] Loading validation datasets...")
    ds_map = load_val_datasets_once(val_csvs)
    print(f"  Loaded {len(ds_map)} validation datasets: {list(ds_map.keys())}")

    print("\n[REPORT] Running final validations...")
    loaded_models = run_final_validations(experiments, logs_root, val_csvs, ds_map)

    if not loaded_models:
        print("[ERROR] No models loaded. Aborting report.")
        return

    print("\n[REPORT] Loading predictions...")
    all_preds = load_experiment_predictions(experiments, logs_root)
    print(f"  Found {len(all_preds)} prediction files")

    print("\n[REPORT] Building summary table...")
    summary_rows = []
    for exp in experiments:
        exp_name = exp["name"]
        run_dir = logs_root / exp_name
        summary_file = run_dir / "summary.json"

        if not summary_file.exists():
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            s = json.load(f)

        row = {
            "experiment": exp_name,
            "agg_arch": exp.get("agg_arch", "?"),
            "lr": exp.get("lr", 0.03),
            "loss_name": exp.get("loss_name", "?"),
            "best_mean_R1": s.get("best_mean_score"),
            "best_epoch": s.get("best_epoch"),
        }

        for ds_name, ds_metrics in s.get("datasets", {}).items():
            short = ds_name.split("/")[0] if "/" in ds_name else ds_name
            for metric_name in ["R1", "R5", "R10"]:
                if metric_name in ds_metrics:
                    row[f"{short}_{metric_name}"] = ds_metrics[metric_name]

        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = report_dir / "comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary saved to {summary_path}")
        plot_metrics_summary(summary_path, report_dir)
    else:
        summary_path = None

    print("\n[REPORT] Finding divergent queries and generating heatmaps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples_dir = report_dir / "query_examples"
    examples_dir.mkdir(exist_ok=True)

    model_names_ordered = sorted(loaded_models.keys())
    all_example_records = []

    for ds_short_name in ds_map:
        print(f"\n  Dataset: {ds_short_name}")

        preds_for_ds = {}
        for exp_name in model_names_ordered:
            key = (exp_name, ds_short_name)
            if key in all_preds:
                preds_for_ds[exp_name] = all_preds[key]

        if len(preds_for_ds) < 2:
            print(f"    [SKIP] Need at least 2 models, have {len(preds_for_ds)}")
            continue

        divergent, consensus = find_divergent_and_consensus(
            preds_for_ds,
            num_divergent=num_divergent,
            num_consensus=num_consensus,
        )

        print(f"    Divergent: {len(divergent)}, Consensus: {len(consensus)}")

        selected = divergent[:num_divergent]
        if len(selected) < num_divergent:
            selected.extend(consensus[: num_divergent - len(selected)])

        if not selected:
            print(f"    [SKIP] No interesting queries found")
            continue

        for idx, q_entry in enumerate(selected):
            q_type = "divergent" if q_entry in divergent else "consensus"
            q_path = q_entry["query_path"]
            q_name = Path(q_path).stem

            ex_dir = (
                examples_dir / f"{ds_short_name}_{q_type}_{idx+1:02d}_{q_name[:40]}"
            )
            ex_dir.mkdir(exist_ok=True)

            query_utm = None
            ref_model = model_names_ordered[0]
            ref_key = (ref_model, ds_short_name)
            if ref_key in all_preds:
                ref_queries = all_preds[ref_key].get("queries", [])

                for q in ref_queries:
                    if q["query_path"] == q_path:
                        qi = q.get("query_idx")
                        if qi is not None and hasattr(ds_map[ds_short_name], "q_utm_np"):
                            q_utm_arr = ds_map[ds_short_name].q_utm_np
                            if qi < len(q_utm_arr):
                                query_utm = q_utm_arr[qi]
                        break

            model_data = {}
            for mname in model_names_ordered:
                if mname not in loaded_models:
                    continue
                # Query attn
                attn_q_dir = ex_dir / f"attn_{mname}"
                attn_q_file = None
                if not attn_q_dir.exists():
                    try:
                        loaded_models[mname].extract_attention_single_image(
                            val_dataset=ds_map[ds_short_name],
                            image_path=q_path,
                            device=device,
                            output_dir=str(attn_q_dir),
                        )
                    except Exception as e:
                        print(f"      [WARN] Query attention failed for {mname}: {e}")
                generated = list(attn_q_dir.glob("*_attention.png"))
                if generated:
                    attn_q_file = str(generated[0])

                # Top-1 attn
                top1_path = q_entry["top1s"].get(mname)
                attn_t1_file = None
                if top1_path and os.path.exists(top1_path):
                    attn_t1_dir = ex_dir / f"attn_{mname}_top1"
                    if not attn_t1_dir.exists():
                        try:
                            loaded_models[mname].extract_attention_single_image(
                                val_dataset=ds_map[ds_short_name],
                                image_path=top1_path,
                                device=device,
                                output_dir=str(attn_t1_dir),
                            )
                        except Exception as e:
                            print(f"      [WARN] Top-1 attention failed for {mname}: {e}")
                    gen_t1 = list(attn_t1_dir.glob("*_attention.png"))
                    if gen_t1:
                        attn_t1_file = str(gen_t1[0])

                model_data[mname] = {
                    "attn_query": attn_q_file,
                    "top1": top1_path,
                    "attn_top1": attn_t1_file,
                }

            top1_distances = {}
            top1_utm_coords = {}
            if query_utm is not None and hasattr(ds_map[ds_short_name], "db_utm_np"):
                db_utm_arr = ds_map[ds_short_name].db_utm_np
                for mname in model_names_ordered:
                    key_m = (mname, ds_short_name)
                    if key_m in all_preds:
                        qs = all_preds[key_m].get("queries", [])
                        for q in qs:
                            if q["query_path"] == q_path:
                                top1_idx = None
                                if "top5_indices" in q and len(q["top5_indices"]) > 0:
                                    top1_idx = q["top5_indices"][0]
                                elif "top5_paths" in q and len(q["top5_paths"]) > 0:
                                    pass
                                if top1_idx is not None and top1_idx < len(db_utm_arr):
                                    top1_utm = db_utm_arr[top1_idx]
                                    dist = np.linalg.norm(query_utm - top1_utm)
                                    top1_distances[mname] = float(dist)
                                    top1_utm_coords[mname] = top1_utm.tolist()
                                else:
                                    top1_distances[mname] = None
                                    top1_utm_coords[mname] = None
                                break

            create_detailed_attention_figure(
                query_path=q_path,
                model_data=model_data,
                model_hits=q_entry["hits"],
                model_names_ordered=model_names_ordered,
                output_path=ex_dir / "detailed_comparison.png",
                title=f"{ds_short_name} | {q_type} #{idx+1}",
            )

            create_multimodel_comparison_figure(
                query_path=q_path,
                model_top1s=q_entry["top1s"],
                model_hits=q_entry["hits"],
                model_names_ordered=model_names_ordered,
                output_path=ex_dir / "comparison_top1.png",
                title=f"{ds_short_name} | {q_type} #{idx+1}",
            )

            simple_attn_paths = {}
            for mname in model_names_ordered:
                if model_data[mname]["attn_query"]:
                    simple_attn_paths[mname] = model_data[mname]["attn_query"]
            if simple_attn_paths:
                create_multimodel_heatmap_figure(
                    query_path=q_path,
                    model_attn_paths=simple_attn_paths,
                    model_names_ordered=model_names_ordered,
                    output_path=ex_dir / "comparison_heatmaps.png",
                    title=f"{ds_short_name} | Attention | {q_type} #{idx+1}",
                )

            try:
                shutil.copy2(q_path, ex_dir / "query.jpg")
            except Exception:
                pass

            # --- Buduj rekord z dodatkowymi informacjami ---
            record = {
                "dataset": ds_short_name,
                "type": q_type,
                "index": idx + 1,
                "query_path": q_path,
                "example_dir": str(ex_dir.relative_to(report_dir)),
            }
            if query_utm is not None:
                record["query_utm_e"] = query_utm[0]
                record["query_utm_n"] = query_utm[1]

            for mname in model_names_ordered:
                record[f"{mname}_hit"] = q_entry["hits"].get(mname, False)
                record[f"{mname}_top1_path"] = q_entry["top1s"].get(mname, "N/A")
                if mname in top1_distances:
                    record[f"{mname}_top1_dist_m"] = top1_distances[mname]
                    if top1_utm_coords.get(mname):
                        record[f"{mname}_top1_utm_e"] = top1_utm_coords[mname][0]
                        record[f"{mname}_top1_utm_n"] = top1_utm_coords[mname][1]
                else:
                    record[f"{mname}_top1_dist_m"] = None

            all_example_records.append(record)

    examples_index_path = report_dir / "query_examples_index.csv"
    if all_example_records:
        pd.DataFrame(all_example_records).to_csv(examples_index_path, index=False)
        print(f"\n  Examples index saved to {examples_index_path}")

    for model in loaded_models.values():
        del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'='*80}")
    print(f"COMPARISON REPORT FINISHED")
    print(f"  Report dir: {report_dir}")
    print(f"  Summary: {summary_path if summary_path else 'N/A'}")
    print(f"  Examples: {examples_dir}")
    print(f"{'='*80}")

def plot_metrics_summary(summary_csv_path: Path, output_dir: Path):
    df = pd.read_csv(summary_csv_path)
    metric_cols = [c for c in df.columns if "_R1" in c or "_R5" in c or "_R10" in c]
    if not metric_cols:
        return

    for metric in metric_cols:
        plt.figure(figsize=(8, 4))
        bars = plt.bar(df["experiment"], df[metric], color="skyblue", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"Porównanie {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"bar_{metric}.png", dpi=150)
        plt.close()

# ====================================================================
#  MAIN
# ====================================================================


def main():
    config = PipelineConfig()
    config.DATAFRAMES_ROOT.mkdir(parents=True, exist_ok=True)

    DATA_CONFIG = [
        # --- Train ---
        {
            "set_type": "train",
            "region_name": "Taizhou-1",
            "uav_visloc_id": "03",
            "map_filename": "satellite03.tif",
            "crop_range_meters": 295,
            "overlap_stride_meters": 195,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Yunnan",
            "uav_visloc_id": "05",
            "map_filename": "satellite05.tif",
            "crop_range_meters": 365,
            "overlap_stride_meters": 265,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Changjiang-20",
            "uav_visloc_id": "01",
            "map_filename": "satellite01.tif",
            "crop_range_meters": 310,
            "overlap_stride_meters": 200,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Taizhou-6",
            "uav_visloc_id": "04",
            "map_filename": "satellite04.tif",
            "crop_range_meters": 315,
            "overlap_stride_meters": 215,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Zhuxi",
            "uav_visloc_id": "06",
            "map_filename": "satellite06.tif",
            "crop_range_meters": 325,
            "overlap_stride_meters": 225,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Huzhou-3",
            "uav_visloc_id": "08",
            "map_filename": "satellite08.tif",
            "crop_range_meters": 320,
            "overlap_stride_meters": 220,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "train",
            "region_name": "Huailai",
            "uav_visloc_id": "10",
            "map_filename": "satellite10.tif",
            "crop_range_meters": 315,
            "overlap_stride_meters": 215,
            "output_suffix": "one_to_one",
        },
        {
            "set_type": "val",
            "region_name": "Changjiang-23",
            "uav_visloc_id": "02",
            "map_filename": "satellite02.tif",
            "crop_range_meters": 310,
            "overlap_stride_meters": 210,
            "val_variant": "v2",
            "output_suffix": "v2_one_to_one",
        },
        {
            "set_type": "val",
            "region_name": "Shandan",
            "uav_visloc_id": "11",
            "map_filename": "satellite11.tif",
            "crop_range_meters": 370,
            "overlap_stride_meters": 270,
            "val_variant": "v2",
            "output_suffix": "v2_one_to_one",
        },
    ]

    # --- Tile Generation ---
    all_csv_paths = {}
    for d_conf in DATA_CONFIG:
        region_name = d_conf["region_name"]
        output_csv_path = config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv"
        all_csv_paths[region_name] = str(output_csv_path)
        thumb_dir = config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR / region_name

        skip_generation = clearup_generated_data(
            config, output_csv_path, thumb_dir, region_name
        )
        if not skip_generation:
            map_tif_path = (
                config.UAV_VISLOC_ROOT
                / d_conf["uav_visloc_id"]
                / d_conf["map_filename"]
            )
            map_sat = MapSatellite(
                csv_path=str(
                    config.UAV_VISLOC_ROOT / "satellite_ coordinates_range.csv"
                ),
                tiles_satellite_csv_output_path=str(output_csv_path),
                map_tif_path=str(map_tif_path),
                region_name=region_name,
                friendly_name=f"visloc-{region_name}-{d_conf['uav_visloc_id']}-satellite",
            )
            thumb_gen = OverlapingTilesGenerator(
                output_dir=str(config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR),
                satellite_map_names=[map_sat],
                crop_range_meters=d_conf["crop_range_meters"],
                overlap_stride_meters=d_conf["overlap_stride_meters"],
                is_rebuild_csv=config.force_regenerate_tiles,
            )
            thumb_gen.generate_tiles()

            uav_gen = UavSmallerCropGenerator(
                csv_path=str(
                    config.UAV_VISLOC_ROOT
                    / d_conf["uav_visloc_id"]
                    / f"{d_conf['uav_visloc_id']}.csv"
                ),
                cropped_uav_csv_output_path=str(output_csv_path),
                cropped_output_dir=str(config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR),
                uav_images_dir=str(
                    config.UAV_VISLOC_ROOT / d_conf["uav_visloc_id"] / "drone"
                ),
                region_name=region_name,
                friendly_name=f"visloc-{region_name}-{d_conf['uav_visloc_id']}-uav",
            )
            uav_gen.generate_tiles()

    # --- Place ID Generation (NOWATER only —sanity check) ---
    train_csvs, val_csvs = generate_place_ids_for_variant(
        config=config,
        data_config=DATA_CONFIG,
        use_informativeness_filter=True,
        uav_overlap_multiplier=1.0,
    )

    print(f"\nTrain CSVs: {len(train_csvs)}")
    print(f"Val CSVs: {len(val_csvs)}")

    experiments = [
    # 1. ResNet50 + SALAD_Resnet, MultiSimilarityLoss
    # {
    #     "seed": 42, "max_epochs": 40, "batch_size": 32,
    #     "loss_name": "MultiSimilarityLoss",
    #     "miner_name": "MultiSimilarityMiner",
    #     "miner_margin": 0.1,
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "lr": 1e-4,
    #     "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_ResNet50_SALADres_MS_s42",
    #     "backbone_arch": "resnet50",
    #     "backbone_config": {},
    #     "agg_arch": "SALAD_Resnet",
    #     "agg_config": {
    #         "num_channels": 2048,
    #         "num_clusters": 32,
    #         "cluster_dim": 128,
    #         "token_dim": 256,
    #     },
    # },
    # # 2. DINOv2 + SALAD, MultiSimilarityLoss
    # {
    #     "seed": 42, "max_epochs": 40, "batch_size": 32,
    #     "loss_name": "MultiSimilarityLoss",
    #     "miner_name": "MultiSimilarityMiner",
    #     "miner_margin": 0.1,
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "lr": 6e-5,
    #     "weight_decay": 9.5e-9,
    #     "lr_sched": "linear",
    #     "lr_sched_args": {
    #         "start_factor": 1,
    #         "end_factor": 0.2,
    #         "total_iters": 4000,
    #     },
    #     "name": "NOWATER_DINOv2_SALAD_MS_s42",
    #     "backbone_arch": "dinov2_vitb14",
    #     "backbone_config": {
    #         "num_trainable_blocks": 4,
    #         "return_token": True,
    #         "norm_layer": True,
    #     },
    #     "agg_arch": "SALAD",
    #     "agg_config": {
    #         "num_channels": 768,
    #         "num_clusters": 64,
    #         "cluster_dim": 128,
    #         "token_dim": 256,
    #     },
    # },
    # # 3. ResNet50 + SALAD_Resnet, TripletMarginLoss (all)
    # {
    #     "seed": 42, "max_epochs": 40, "batch_size": 32,
    #     "loss_name": "TripletMarginLoss",
    #     "miner_name": "TripletMarginMiner",
    #     "loss_margin": 0.05,
    #     "miner_margin": 0.05,
    #     "type_of_triplets": "all",
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "swap": False, "smooth_loss": False,
    #     "lr": 1e-4,
    #     "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_ResNet50_SALADres_TripletAll_s42",
    #     "backbone_arch": "resnet50",
    #     "backbone_config": {},
    #     "agg_arch": "SALAD_Resnet",
    #     "agg_config": {
    #         "num_channels": 2048,
    #         "num_clusters": 32,
    #         "cluster_dim": 128,
    #         "token_dim": 256,
    #     },
    # },
    # 4. GeM, all, s42
    # {
    #     "seed": 42, "max_epochs": 40, "batch_size": 32,
    #     "loss_name": "TripletMarginLoss",
    #     "miner_name": "TripletMarginMiner",
    #     "loss_margin": 0.05,
    #     "miner_margin": 0.05,
    #     "type_of_triplets": "all",
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "swap": False, "smooth_loss": False,
    #     "lr": 1e-4, "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_GeM_all_s42",
    #     "agg_arch": "GeM",
    #     "agg_config": {"p": 3, "eps": 1e-6},
    # },
    # # 5. GeM, semihard, s42
    # {
    #     "seed": 42, "max_epochs": 40, "batch_size": 32,
    #     "loss_name": "TripletMarginLoss",
    #     "miner_name": "TripletMarginMiner",
    #     "loss_margin": 0.05,
    #     "miner_margin": 0.05,
    #     "type_of_triplets": "semihard",
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "swap": False, "smooth_loss": False,
    #     "lr": 1e-4, "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_GeM_semihard_s42",
    #     "agg_arch": "GeM",
    #     "agg_config": {"p": 3, "eps": 1e-6},
    # },
    # 6. GeM, all, s123, bs64 
    # {
    #     "seed": 123, "max_epochs": 40, "batch_size": 64,
    #     "loss_name": "TripletMarginLoss",
    #     "miner_name": "TripletMarginMiner",
    #     "loss_margin": 0.05,
    #     "miner_margin": 0.05,
    #     "type_of_triplets": "all",
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "swap": False, "smooth_loss": False,
    #     "lr": 1e-4, "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_GeM_all_s123_bs64",
    #     "agg_arch": "GeM",
    #     "agg_config": {"p": 3, "eps": 1e-6},
    # },
    # # 7. GeM, semihard, s123, bs64
    # {
    #     "seed": 123, "max_epochs": 40, "batch_size": 64,
    #     "loss_name": "TripletMarginLoss",
    #     "miner_name": "TripletMarginMiner",
    #     "loss_margin": 0.05,
    #     "miner_margin": 0.05,
    #     "type_of_triplets": "semihard",
    #     "distance": "CosineSimilarity",
    #     "optimizer": "adamw",
    #     "swap": False, "smooth_loss": False,
    #     "lr": 1e-4, "lr_sched": "cosine",
    #     "lr_sched_args": {"T_max": 35},
    #     "name": "NOWATER_GeM_semihard_s123_bs64",
    #     "agg_arch": "GeM",
    #     "agg_config": {"p": 3, "eps": 1e-6},
    # },
    # 8. ConvAP, all, s123
    {
        "seed": 123, "max_epochs": 40, "batch_size": 32,
        "loss_name": "TripletMarginLoss",
        "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05,
        "miner_margin": 0.05,
        "type_of_triplets": "all",
        "distance": "CosineSimilarity",
        "optimizer": "adamw",
        "swap": False, "smooth_loss": False,
        "lr": 1e-4, "lr_sched": "cosine",
        "lr_sched_args": {"T_max": 35},
        "name": "NOWATER_ConvAP_all_s123",
        "agg_arch": "ConvAP",
        "agg_config": {
            "in_channels": 2048,
            "out_channels": 512,
            "s1": 2,
            "s2": 2,
        },
    },
    # 9. GeM, ContrastiveLoss + MultiSimilarityMiner
    {
        "seed": 42, "max_epochs": 40, "batch_size": 32,
        "loss_name": "ContrastiveLoss",
        "miner_name": "MultiSimilarityMiner",
        "loss_margin": 0.8,
        "loss_margin_neg": 0.4,
        "miner_margin": 0.1,
        "distance": "CosineSimilarity",
        "optimizer": "adamw",
        "lr": 1e-4, "lr_sched": "cosine",
        "lr_sched_args": {"T_max": 35},
        "name": "NOWATER_GeM_ContrastiveMS_s42",
        "agg_arch": "GeM",
        "agg_config": {"p": 3, "eps": 1e-6},
    },
]

    # --- Training ---
    logs_root = Path("./logs_compare").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    SKIP_TRAINING = False
    if not SKIP_TRAINING:
        for exp in experiments:
            result = run_single_experiment(exp, train_csvs, val_csvs, logs_root)

            run_dir = logs_root / exp["name"]
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            all_results.append(result)

            pd.DataFrame(all_results).to_csv(logs_root / "all_results.csv", index=False)
    else:
        print("Skipping training, using existing results.")

    # --- Comparison Report ---
    print("\n" + "=" * 100)
    print("GENERATING COMPARISON REPORT")
    print("=" * 100)

    generate_comparison_report(
        experiments=experiments,
        logs_root=logs_root,
        val_csvs=val_csvs,
        num_divergent=10,
        num_consensus=5,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
