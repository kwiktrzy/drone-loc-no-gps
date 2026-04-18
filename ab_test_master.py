from typing import List
import gc
import json
import os
import shutil
import inspect
import copy
from pathlib import Path
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUSI być przed importem pyplot
import matplotlib.pyplot as plt
from PIL import Image

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.place_id_generators.ManyToManyPlaceIdGenerator import ManyToManyPlaceIdGenerator
from dataset_splitter.structs.MapSatellite import MapSatellite
from dataset_splitter.satellite_generators.OverlapingTilesGenerator import OverlapingTilesGenerator
from dataset_splitter.uav_generators.UavSmallerCropGenerator import UavSmallerCropGenerator
from vpr_model import VPRModel


class PipelineConfig:
    def __init__(self, project_root="/workspace/"):
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

        self.force_regenerate_tiles = False
        self.force_regenerate_place_ids = False
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

    checkpoint_min = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_min_R1_4sets",
        filename="best_min-{epoch:02d}-{val_min_R1_4sets:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode="max",
    )

    checkpoint_shandan_v1 = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="Shandan-v1_one_to_one/R1",
        filename="best_shandan_v1-{epoch:02d}-{Shandan-v1_one_to_one/R1:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode="max",
    )

    checkpoint_changjiang_v1 = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="Changjiang-23-v1_one_to_one/R1",
        filename="best_changjiang_v1-{epoch:02d}-{Changjiang-23-v1_one_to_one/R1:.4f}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        mode="max",
    )

    callbacks = [
        checkpoint_mean,
        checkpoint_min,
        checkpoint_shandan_v1,
        checkpoint_changjiang_v1,
    ]

    cb_map = {
        "mean": checkpoint_mean,
        "min": checkpoint_min,
        "shandan_v1": checkpoint_shandan_v1,
        "changjiang_v1": checkpoint_changjiang_v1,
    }
    return callbacks, cb_map


def score_to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def ensure_place_ids_for_variant(
    config: PipelineConfig,
    data_config: List[dict],
    use_water_removal: bool,
    force_regenerate: bool = False
) -> tuple[List[str], List[str]]:
    water_tag = "water" if use_water_removal else "nowater"
    train_csvs: List[str] = []
    val_csvs: List[str] = []

    print(f"\n=== Ensuring Place IDs for variant: {water_tag.upper()} ===")

    for d_conf in data_config:
        region_name = d_conf["region_name"]
        base_path = str(config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv")
        final_suffix = f"{d_conf['output_suffix']}_{water_tag}"
        final_path = get_processed_path(base_path, final_suffix)

        is_val = d_conf["set_type"] == "val"
        need_generate = force_regenerate or not Path(final_path).exists()

        if need_generate:
            print(f"  [GEN] {region_name} -> {final_path}")
            generator = ManyToManyPlaceIdGenerator(
                csv_tiles_path=base_path,
                csv_place_ids_output_path=final_path,
                force_regenerate=True,
                is_validation_set=is_val,
                is_validation_set_v2=d_conf.get("val_variant") == "v2",
                radius_neighbors_meters=70 if is_val else d_conf["crop_range_meters"],
                tiles_trash_directory=config.DATAFRAMES_TILES_TRASH,
                use_informativeness_filter=use_water_removal,  # KLUCZOWE
            )
            generator.generate_place_ids()
        else:
            print(f"  [SKIP] {region_name} (already exists)")

        if d_conf["set_type"] == "train":
            train_csvs.append(final_path)
        elif d_conf["set_type"] == "val":
            val_csvs.append(final_path)

    return train_csvs, val_csvs


def run_experiment(
    exp: dict,
    train_csvs: List[str],
    val_csvs: List[str],
    logs_root: Path,
    config: PipelineConfig,
    extract_attention: bool = True,
    num_attention_queries: int = 3,
) -> dict:
    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {exp['name']}")
    print(f"  Seed: {exp['seed']}")
    print(f"  Water removal: {exp.get('use_water_removal', True)}")
    print("=" * 100)

    pl.seed_everything(exp["seed"], workers=True)

    run_dir = (logs_root / exp["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "experiment_config.json", "w", encoding='utf-8') as f:
        json.dump(exp, f, indent=2)

    datamodule = MapsDataModule(
        tiles_csv_file_paths=train_csvs,
        batch_size=exp.get("batch_size", 32),
        val_set_names=val_csvs,
        shuffle_all=True
    )

    valid_model_args = inspect.signature(VPRModel.__init__).parameters.keys()
    model_kwargs = {k: v for k, v in exp.items() if k in valid_model_args}
    model = VPRModel(**model_kwargs)

    # AB test hooks
    model.save_val_predictions = True
    model.run_dir = str(run_dir)

    callbacks, cb_map = build_callbacks(run_dir)

    old_cwd = os.getcwd()
    os.chdir(run_dir)
    trainer = None

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

        full_model_path = run_dir / "full_model_final.pth"
        torch.save(model.state_dict(), full_model_path)
        print(f"Saved final model state_dict to: {full_model_path}")

        if extract_attention:
            device = next(model.parameters()).device
            val_dataloaders = trainer.val_dataloaders if hasattr(trainer, 'val_dataloaders') else None
            if val_dataloaders is None:
                val_dataloaders = datamodule.val_dataloader()
            if not isinstance(val_dataloaders, list):
                val_dataloaders = [val_dataloaders]

            for i, (val_dl, val_name) in enumerate(zip(val_dataloaders, val_csvs)):
                short_name = Path(val_name).stem
                attn_dir = run_dir / "attention_maps" / short_name
                val_dataset = datamodule.val_datasets[i]
                try:
                    model.extract_attention_maps(
                        val_dataset=val_dataset,
                        val_dataloader=val_dl,
                        device=device,
                        output_dir=str(attn_dir),
                        num_queries=num_attention_queries,
                    )
                except Exception as e:
                    print(f"[WARN] Attention extraction failed for {short_name}: {e}")

    finally:
        os.chdir(old_cwd)

    metrics_summary = {
        "experiment": exp["name"],
        "seed": exp["seed"],
        "use_water_removal": exp.get("use_water_removal", True),
        "best_mean_score": score_to_float(cb_map["mean"].best_model_score),
        "best_mean_path": cb_map["mean"].best_model_path,
        "best_min_score": score_to_float(cb_map["min"].best_model_score),
        "best_min_path": cb_map["min"].best_model_path,
        "best_shandan_v1_score": score_to_float(cb_map["shandan_v1"].best_model_score),
        "best_changjiang_v1_score": score_to_float(cb_map["changjiang_v1"].best_model_score),
        "final_model_path": str(run_dir / "full_model_final.pth"),
        "run_dir": str(run_dir),
    }

    metrics_jsonl = run_dir / "val_metrics.jsonl"
    if metrics_jsonl.exists():
        with open(metrics_jsonl, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            last_epoch = lines[-1]
            metrics_summary["last_epoch_metrics"] = last_epoch.get("datasets", {})

    del trainer
    del model
    del datamodule
    torch.cuda.empty_cache()
    gc.collect()

    return metrics_summary

def select_interesting_queries(predictions_water: dict, predictions_nowater: dict, 
                                 num_divergent: int = 3, num_consensus: int = 2) -> tuple:
    """
    Selects interesting queries for visualization.
    
    Returns:
        divergent: list where water and no-water disagree (priority: water=HIT, no-water=MISS)
        consensus_hits: list where both agree and both are HIT (for comparison)
    """
    divergent = []
    consensus_hits = []
    
    queries_w = predictions_water.get("queries", [])
    queries_n = predictions_nowater.get("queries", [])
    
    if len(queries_w) != len(queries_n):
        print(f"[WARN] Query count mismatch: water={len(queries_w)}, no-water={len(queries_n)}")
        return [], []
    
    for qw, qn in zip(queries_w, queries_n):
        if qw.get("query_path") != qn.get("query_path"):
            continue  # sanity check failed
            
        hit_w = qw.get("is_hit_r1", False)
        hit_n = qn.get("is_hit_r1", False)
        
        entry = {
            "query_path": qw["query_path"],
            "water_hit": hit_w,
            "nowater_hit": hit_n,
            "water_top1": qw.get("top5_paths", [None])[0] if qw.get("top5_paths") else None,
            "nowater_top1": qn.get("top5_paths", [None])[0] if qn.get("top5_paths") else None,
            "water_top5": qw.get("top5_paths", []),
            "nowater_top5": qn.get("top5_paths", []),
        }
        
        if hit_w != hit_n:
            divergent.append(entry)
        elif hit_w and hit_n:  # both hit
            consensus_hits.append(entry)
            
        if len(divergent) >= num_divergent and len(consensus_hits) >= num_consensus:
            break
    
    # If not enough divergent, add some consensus hits as filler
    if len(divergent) < num_divergent:
        print(f"[INFO] Only {len(divergent)} divergent queries found, adding consensus hits")
        
    return divergent[:num_divergent], consensus_hits[:num_consensus]

def create_comparison_figure(query_path: str, water_top1: str, nowater_top1: str,
                             output_path: Path, title: str = ""):
    """Creates side-by-side comparison of query and top-1 predictions."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Query
        img_q = Image.open(query_path).convert('RGB')
        axes[0].imshow(img_q)
        axes[0].set_title("Query (UAV)")
        axes[0].axis('off')
        
        # Water variant result
        if water_top1 and os.path.exists(water_top1):
            img_w = Image.open(water_top1).convert('RGB')
            axes[1].imshow(img_w)
            status = "HIT" if "hit" in title.lower() else "Result"
            axes[1].set_title(f"With Water Removal\n{status}")
        else:
            axes[1].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[1].set_title("With Water Removal")
        axes[1].axis('off')
        
        # No-water variant result
        if nowater_top1 and os.path.exists(nowater_top1):
            img_n = Image.open(nowater_top1).convert('RGB')
            axes[2].imshow(img_n)
            status = "MISS" if "miss" in title.lower() else "Result"
            axes[2].set_title(f"Without Water Removal\n{status}")
        else:
            axes[2].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[2].set_title("Without Water Removal")
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create figure: {e}")
        return False

def generate_ab_report(logs_root: Path, output_dir: Path, num_examples: int = 3):
    """
    Generates final AB test report: tables, charts, and query/result examples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = logs_root / "ab_results.csv"
    
    if not results_csv.exists():
        print("No ab_results.csv found. Skipping report.")
        return

    df = pd.read_csv(results_csv)
    
    # Parse metrics
    rows = []
    for _, row in df.iterrows():
        base = {
            "experiment": row["experiment"],
            "seed": row["seed"],
            "use_water_removal": row["use_water_removal"],
            "best_mean_R1": row["best_mean_score"],
        }
        metrics_json = row.get("last_epoch_metrics", "{}")
        if isinstance(metrics_json, str):
            try:
                metrics_json = json.loads(metrics_json)
            except Exception:
                metrics_json = {}
        for ds_name, vals in metrics_json.items():
            rows.append({
                **base,
                "dataset": ds_name,
                "R1": vals.get("R1", None),
                "R5": vals.get("R5", None),
                "R10": vals.get("R10", None),
            })
    
    report_df = pd.DataFrame(rows)
    report_path = output_dir / "ab_report_table.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Report table saved to {report_path}")

    # Statistical summary per dataset
    summary_rows = []
    for ds_name in report_df["dataset"].unique():
        ds_data = report_df[report_df["dataset"] == ds_name]
        water_data = ds_data[ds_data["use_water_removal"] == True]
        nowater_data = ds_data[ds_data["use_water_removal"] == False]
        
        if len(water_data) > 0 and len(nowater_data) > 0:
            water_r1_mean = water_data["R1"].mean()
            nowater_r1_mean = nowater_data["R1"].mean()
            improvement = water_r1_mean - nowater_r1_mean
            
            summary_rows.append({
                "dataset": ds_name,
                "water_r1_mean": water_r1_mean,
                "nowater_r1_mean": nowater_r1_mean,
                "absolute_improvement": improvement,
                "relative_improvement_pct": (improvement / nowater_r1_mean * 100) if nowater_r1_mean > 0 else 0,
                "num_seeds": min(len(water_data), len(nowater_data)),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "ab_summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path}")

    # Bar chart comparison
    if not report_df.empty:
        metrics = ["R1", "R5", "R10"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, metric in zip(axes, metrics):
            if report_df[metric].isna().all():
                ax.set_visible(False)
                continue
                
            agg = report_df.groupby(["dataset", "use_water_removal"])[metric].agg(['mean', 'std']).reset_index()
            datasets = agg["dataset"].unique()
            x = np.arange(len(datasets))
            width = 0.35
            
            water_means = agg[agg["use_water_removal"] == True].set_index("dataset")["mean"]
            nowater_means = agg[agg["use_water_removal"] == False].set_index("dataset")["mean"]
            water_std = agg[agg["use_water_removal"] == True].set_index("dataset")["std"]
            nowater_std = agg[agg["use_water_removal"] == False].set_index("dataset")["std"]
            
            vals_water = [water_means.get(d, 0) for d in datasets]
            vals_nowater = [nowater_means.get(d, 0) for d in datasets]
            err_water = [water_std.get(d, 0) for d in datasets]
            err_nowater = [nowater_std.get(d, 0) for d in datasets]
            
            bars1 = ax.bar(x - width/2, vals_nowater, width, yerr=err_nowater,
                          label='Baseline (no water removal)', capsize=3, color='coral', alpha=0.8)
            bars2 = ax.bar(x + width/2, vals_water, width, yerr=err_water,
                          label='With water removal', capsize=3, color='seagreen', alpha=0.8)
            
            # Value labels
            for bar in bars1:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel(f'Recall@{metric[-1]}', fontsize=11)
            ax.set_title(f'Recall@{metric[-1]} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(max(vals_water, default=0), max(vals_nowater, default=0)) * 1.2)
        
        fig.suptitle('AB Test: Impact of Water Removal on VPR Performance', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "ab_comparison_all_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to {output_dir / 'ab_comparison_all_metrics.png'}")

    # Query/Result Examples with Visualizations
    examples_dir = output_dir / "query_examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Collect all predictions
    all_predictions = {}
    for _, row in df.iterrows():
        run_dir = Path(row["run_dir"])
        pred_files = list(run_dir.glob("predictions_*.json"))
        for pf in pred_files:
            ds_name = pf.stem.replace("predictions_", "")
            key = (row["seed"], row["use_water_removal"], ds_name)
            try:
                with open(pf, 'r', encoding='utf-8') as f:
                    all_predictions[key] = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {pf}: {e}")

    # Generate examples per seed and dataset
    example_records = []
    seeds = sorted(df["seed"].unique())
    
    for seed in seeds:
        for ds_name in ["Shandan-v1_one_to_one", "Changjiang-23-v1_one_to_one"]:
            key_water = (seed, True, ds_name)
            key_nowater = (seed, False, ds_name)
            
            if key_water not in all_predictions or key_nowater not in all_predictions:
                continue
            
            preds_w = all_predictions[key_water]
            preds_n = all_predictions[key_nowater]
            
            divergent, consensus = select_interesting_queries(preds_w, preds_n, 
                                                              num_divergent=num_examples, 
                                                              num_consensus=2)
            
            # Process divergent examples (most interesting)
            for idx, ex in enumerate(divergent):
                ex_dir = examples_dir / f"seed{seed}_{ds_name}_ex{idx+1:02d}_{'water_hit' if ex['water_hit'] else 'water_miss'}"
                ex_dir.mkdir(exist_ok=True)
                
                # Copy images
                try:
                    if os.path.exists(ex["query_path"]):
                        shutil.copy2(ex["query_path"], ex_dir / "query.jpg")
                    if ex["water_top1"] and os.path.exists(ex["water_top1"]):
                        shutil.copy2(ex["water_top1"], ex_dir / "water_top1.jpg")
                    if ex["nowater_top1"] and os.path.exists(ex["nowater_top1"]):
                        shutil.copy2(ex["nowater_top1"], ex_dir / "nowater_top1.jpg")
                except Exception as e:
                    print(f"[WARN] Could not copy images for example: {e}")
                
                # Create comparison figure
                title = f"{ds_name} | Seed {seed}\nWater: {'HIT' if ex['water_hit'] else 'MISS'} | No-Water: {'HIT' if ex['nowater_hit'] else 'MISS'}"
                create_comparison_figure(
                    ex["query_path"], ex["water_top1"], ex["nowater_top1"],
                    ex_dir / "comparison.png", title=title
                )
                
                # Record for text report
                example_records.append({
                    "seed": seed,
                    "dataset": ds_name,
                    "example_num": idx + 1,
                    "query_path": ex["query_path"],
                    "water_result": "HIT" if ex["water_hit"] else "MISS",
                    "nowater_result": "HIT" if ex["nowater_hit"] else "MISS",
                    "improvement": "YES" if (ex["water_hit"] and not ex["nowater_hit"]) else "NO",
                    "example_dir": str(ex_dir.relative_to(output_dir))
                })
            
            # If no divergent found, use consensus hits
            if not divergent and consensus:
                print(f"[INFO] No divergent queries for {ds_name} seed {seed}, showing consensus HIT")
                for idx, ex in enumerate(consensus[:2]):
                    ex_dir = examples_dir / f"seed{seed}_{ds_name}_consensus{idx+1:02d}"
                    ex_dir.mkdir(exist_ok=True)
                    # ... (similar processing, omitted for brevity)

    # Generate text report
    examples_path = output_dir / "ab_query_examples.txt"
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AB TEST: QUERY/RESULT EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        if summary_df is not None and not summary_df.empty:
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"\nDataset: {row['dataset']}\n")
                f.write(f"  Baseline (no water):  R1 = {row['nowater_r1_mean']:.4f}\n")
                f.write(f"  With water removal:   R1 = {row['water_r1_mean']:.4f}\n")
                f.write(f"  Absolute improvement: {row['absolute_improvement']:+.4f}\n")
                f.write(f"  Relative improvement: {row['relative_improvement_pct']:+.2f}%\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Key findings
        improvements = [r for r in example_records if r["improvement"] == "YES"]
        if improvements:
            f.write("KEY FINDINGS: Cases where water removal IMPROVED Recall@1\n")
            f.write("-" * 60 + "\n")
            f.write("These are cases where the baseline (no water removal) failed to\n")
            f.write("retrieve the correct match at rank 1, but the model trained with\n")
            f.write("water removal succeeded. This suggests that removing water helps\n")
            f.write("the model learn more discriminative features.\n\n")
            
            for rec in improvements[:5]:  # Top 5
                f.write(f"\nExample: {rec['example_dir']}\n")
                f.write(f"  Seed: {rec['seed']}, Dataset: {rec['dataset']}\n")
                f.write(f"  Query: {rec['query_path']}\n")
                f.write(f"  Result: Baseline={rec['nowater_result']}, Water-removed={rec['water_result']}\n")
                f.write(f"  Interpretation: Removing water enabled correct retrieval.\n")
        
        # Full listing
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("FULL LIST OF EXAMPLES\n")
        f.write("=" * 80 + "\n")
        for rec in example_records:
            f.write(f"\n{rec['example_dir']}\n")
            f.write(f"  Seed {rec['seed']} | {rec['dataset']}\n")
            f.write(f"  Query: {Path(rec['query_path']).name}\n")
            f.write(f"  Water removal: {rec['water_result']} | Baseline: {rec['nowater_result']}\n")
    
    print(f"Examples report saved to {examples_path}")
    print(f"Visual examples saved to {examples_dir}")
    
    # Final summary print
    print("\n" + "=" * 80)
    print("AB TEST REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")


def main():
    config = PipelineConfig()
    config.force_regenerate_place_ids = True
    config.DATAFRAMES_ROOT.mkdir(parents=True, exist_ok=True)

    DATA_CONFIG = [
        {
            "set_type": "train",
            "region_name": "Taizhou-1",
            "uav_visloc_id": "03",
            "map_filename": "satellite03.tif",
            "crop_range_meters": 295,
            "overlap_stride_meters": 195,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Yunnan",
            "uav_visloc_id": "05",
            "map_filename": "satellite05.tif",
            "crop_range_meters": 365,
            "overlap_stride_meters": 265,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Changjiang-20",
            "uav_visloc_id": "01",
            "map_filename": "satellite01.tif",
            "crop_range_meters": 310,
            "overlap_stride_meters": 200,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "val",
            "region_name": "Changjiang-23",
            "uav_visloc_id": "02",
            "map_filename": "satellite02.tif",
            "crop_range_meters": 310,
            "overlap_stride_meters": 200,
            "val_variant": "v1",
            "output_suffix": "v1_one_to_one"
        },
        {
            "set_type": "val",
            "region_name": "Changjiang-23",
            "uav_visloc_id": "02",
            "map_filename": "satellite02.tif",
            "crop_range_meters": 310,
            "overlap_stride_meters": 210,
            "val_variant": "v2",
            "output_suffix": "v2_one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Taizhou-6",
            "uav_visloc_id": "04",
            "map_filename": "satellite04.tif",
            "crop_range_meters": 315,
            "overlap_stride_meters": 215,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Zhuxi",
            "uav_visloc_id": "06",
            "map_filename": "satellite06.tif",
            "crop_range_meters": 325,
            "overlap_stride_meters": 225,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Huzhou-3",
            "uav_visloc_id": "08",
            "map_filename": "satellite08.tif",
            "crop_range_meters": 320,
            "overlap_stride_meters": 220,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "train",
            "region_name": "Huailai",
            "uav_visloc_id": "10",
            "map_filename": "satellite10.tif",
            "crop_range_meters": 315,
            "overlap_stride_meters": 215,
            "output_suffix": "one_to_one"
        },
        {
            "set_type": "val",
            "region_name": "Shandan",
            "uav_visloc_id": "11",
            "map_filename": "satellite11.tif",
            "crop_range_meters": 370,
            "overlap_stride_meters": 270,
            "val_variant": "v1",
            "output_suffix": "v1_one_to_one"
        },
        {
            "set_type": "val",
            "region_name": "Shandan",
            "uav_visloc_id": "11",
            "map_filename": "satellite11.tif",
            "crop_range_meters": 370,
            "overlap_stride_meters": 270,
            "val_variant": "v2",
            "output_suffix": "v2_one_to_one"
        },
    ]

    all_csv_paths_one_to_one = {}
    for d_conf in DATA_CONFIG:
        region_name = d_conf["region_name"]
        if config.one_to_one_tiles:
            output_csv_path = config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv"
            all_csv_paths_one_to_one[region_name] = str(output_csv_path)
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
                    csv_path=str(config.UAV_VISLOC_ROOT / "satellite_ coordinates_range.csv"),
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
                    csv_path=str(config.UAV_VISLOC_ROOT / d_conf["uav_visloc_id"] / f"{d_conf['uav_visloc_id']}.csv"),
                    cropped_uav_csv_output_path=str(output_csv_path),
                    cropped_output_dir=str(config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR),
                    uav_images_dir=str(config.UAV_VISLOC_ROOT / d_conf["uav_visloc_id"] / "drone"),
                    region_name=region_name,
                    friendly_name=f"visloc-{region_name}-{d_conf['uav_visloc_id']}-uav",
                )
                uav_gen.generate_tiles()

    base_exp = {
        "max_epochs": 40,
        "loss_name": "TripletMarginLoss",
        "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05,
        "miner_margin": 0.05,
        "type_of_triplets": "all",
        "swap": False,
        "smooth_loss": False,
        "agg_arch": "GeM",
        "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "warmup_cosine",
        "lr_sched_args": {
            "warmup_fraction": 0.05,
            "eta_min_ratio": 0.01,
        },
    }

    seeds = [42, 123, 999]
    
    # Helper to avoid repetition
    def make_experiments_for_config(base_name, cfg, seeds):
        exps = []
        for seed in seeds:
            exps.append({**cfg, "name": f"{base_name}_WATER_s{seed}",  "seed": seed, "use_water_removal": True})
            exps.append({**cfg, "name": f"{base_name}_NOWATER_s{seed}", "seed": seed, "use_water_removal": False})
        return exps

    # --- CONFIG 1: Contrastive + PairMargin + GeM + multistep ---
    cfg1 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "ContrastiveLoss", "miner_name": "PairMarginMiner",
        "loss_margin": 0.8, "loss_margin_neg": 0.4,
        "miner_margin": 0.8, "miner_margin_neg": 0.6,
        "distance": "DotProductSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "multistep", "lr_sched_args": {"milestones": [20, 30], "gamma": 0.1},
    }

    # --- CONFIG 2: Contrastive + MSMiner + GeM + cosine ---
    cfg2 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "ContrastiveLoss", "miner_name": "MultiSimilarityMiner",
        "loss_margin": 0.8, "loss_margin_neg": 0.5,
        "miner_margin": 0.12,
        "distance": "CosineSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    # --- CONFIG 3: Triplet + TripletMiner + GeM + cosine ---
    cfg3 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "TripletMarginLoss", "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05, "miner_margin": 0.05,
        "type_of_triplets": "all", "swap": False, "smooth_loss": False,
        "distance": "CosineSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    # --- CONFIG 4: Triplet + TripletMiner + ConvAP + cosine ---
    cfg4 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "TripletMarginLoss", "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05, "miner_margin": 0.05,
        "type_of_triplets": "all", "swap": False, "smooth_loss": False,
        "distance": "CosineSimilarity",
        "agg_arch": "ConvAP", "agg_config": {"in_channels": 2048, "out_channels": 512, "s1": 2, "s2": 2},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    EXPERIMENTS = []
    EXPERIMENTS += make_experiments_for_config("C1_ContrastivePair_GeM", cfg1, seeds)
    EXPERIMENTS += make_experiments_for_config("C2_ContrastiveMS_GeM", cfg2, seeds)
    EXPERIMENTS += make_experiments_for_config("C3_TripletTriplet_GeM", cfg3, seeds)
    EXPERIMENTS += make_experiments_for_config("C4_TripletTriplet_ConvAP", cfg4, seeds)

    print(f"Total experiments: {len(EXPERIMENTS)}")

    logs_root = Path("./logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    
    for exp in EXPERIMENTS:
        train_csvs, val_csvs = ensure_place_ids_for_variant(
            config=config,
            data_config=DATA_CONFIG,
            use_water_removal=exp["use_water_removal"],
            force_regenerate=config.force_regenerate_place_ids,
        )
        
        result = run_experiment(
            exp=exp,
            train_csvs=train_csvs,
            val_csvs=val_csvs,
            logs_root=logs_root,
            config=config,
            extract_attention=True,
            num_attention_queries=3,
        )
        all_results.append(result)
        
        pd.DataFrame(all_results).to_csv(logs_root / "ab_results.csv", index=False)
        print(f"\nIntermediate summary saved to {logs_root / 'ab_results.csv'}")

    print("\n" + "=" * 100)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 100)

    generate_ab_report(
        logs_root=logs_root, 
        output_dir=logs_root / "report",
        num_examples=3
    )


if __name__ == "__main__":
    main()