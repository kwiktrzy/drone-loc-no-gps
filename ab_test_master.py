from typing import List
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
matplotlib.use('Agg')
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


def build_callbacks(run_dir: Path, val_csvs: List[str] = None):
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

    callbacks = [checkpoint_mean, checkpoint_min]
    cb_map = {
        "mean": checkpoint_mean,
        "min": checkpoint_min,
    }

    if val_csvs:
        for csv_path in val_csvs:
            stem = Path(csv_path).stem
            monitor_name = f"{stem}/R1"

            if "Shandan" in stem and "v1" in stem:
                checkpoint = pl.callbacks.ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    monitor=monitor_name,
                    filename="best_shandan_v1-{epoch:02d}",
                    auto_insert_metric_name=False,
                    save_weights_only=True,
                    save_top_k=1,
                    save_last=False,
                    mode="max",
                )
                callbacks.append(checkpoint)
                cb_map["shandan_v1"] = checkpoint

            elif "Changjiang-23" in stem and "v1" in stem:
                checkpoint = pl.callbacks.ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    monitor=monitor_name,
                    filename="best_changjiang_v1-{epoch:02d}",
                    auto_insert_metric_name=False,
                    save_weights_only=True,
                    save_top_k=1,
                    save_last=False,
                    mode="max",
                )
                callbacks.append(checkpoint)
                cb_map["changjiang_v1"] = checkpoint

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
    use_water_removal: bool,  # True = NO_WATER (usuwamy wodę), False = WATER (zostawiamy wodę)
    force_regenerate: bool = False
) -> tuple[List[str], List[str]]:

    water_tag = "water_removed" if use_water_removal else "water_kept"
    train_csvs: List[str] = []
    val_csvs: List[str] = []

    print(f"\n=== Ensuring Place IDs for variant: {water_tag.upper()} ===")

    for d_conf in data_config:
        region_name = d_conf["region_name"]
        base_path = str(config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv")
        
        is_val = d_conf["set_type"] == "val"
        
        # Validation sets always NO_WATER
        if is_val:
            final_suffix = f"{d_conf['output_suffix']}_FIXED_NO_WATER"
            use_filter_for_this = True
            water_tag_for_file = "fixed_no_water"
        else:
            final_suffix = f"{d_conf['output_suffix']}_{water_tag}"
            use_filter_for_this = use_water_removal
            water_tag_for_file = water_tag
        
        final_path = get_processed_path(base_path, final_suffix)
        
        print(f"  Region: {region_name}")
        print(f"    Type: {'VAL' if is_val else 'TRAIN'}")
        print(f"    Filter enabled: {use_filter_for_this}")
        print(f"    Output: {final_path}")
        
        need_generate = force_regenerate or not Path(final_path).exists()
        
        if need_generate:
            print(f"    [GENERATING...]")
            generator = ManyToManyPlaceIdGenerator(
                csv_tiles_path=base_path,
                csv_place_ids_output_path=final_path,
                force_regenerate=True,
                is_validation_set=is_val,
                is_validation_set_v2=d_conf.get("val_variant") == "v2",
                radius_neighbors_meters=70 if is_val else d_conf["crop_range_meters"],
                tiles_trash_directory=config.DATAFRAMES_TILES_TRASH,
                use_informativeness_filter=use_filter_for_this,
            )
            generator.generate_place_ids()
        else:
            print(f"    [EXISTS]")
        
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
) -> dict:
    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {exp['name']}")
    print(f"  Seed: {exp['seed']}")
    print(f"  Water removal: {exp.get('use_water_removal', True)}")
    print("=" * 100)

    pl.seed_everything(exp["seed"], workers=True)

    print("\n[STATS] Training datasets summary:")
    total_uav = 0
    total_sat = 0
    total_places = 0
    for csv_path in train_csvs:
        df = pd.read_csv(csv_path)
        uav_count = len(df[df['friendly-name'].str.contains('uav', case=False, na=False)])
        sat_count = len(df[df['friendly-name'].str.contains('satellite', case=False, na=False)])
        place_count = df['place_id'].nunique()
        
        total_uav += uav_count
        total_sat += sat_count
        total_places += place_count
        
        print(f"  {Path(csv_path).name}:")
        print(f"    UAV: {uav_count}, SAT: {sat_count}, Places: {place_count}")
    
    print(f"\n  TOTAL -> UAV: {total_uav}, SAT: {total_sat}, Places: {total_places}")
    print("=" * 100)

    run_dir = (logs_root / exp["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = run_dir / "val_metrics.jsonl"
    if metrics_jsonl.exists():
        print(f"[CLEAN] Removing stale metrics file: {metrics_jsonl}")
        metrics_jsonl.unlink()

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

    model.save_val_predictions = False
    model.run_dir = str(run_dir)

    callbacks, cb_map = build_callbacks(run_dir, val_csvs=val_csvs)

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

        best_model_path = cb_map["mean"].best_model_path
        if best_model_path and Path(best_model_path).exists():
            print(f"\n[INFO] Wczytywanie globalnie najlepszego modelu z: {best_model_path}")
            model = VPRModel.load_from_checkpoint(best_model_path, strict=True)
        else:
            print("[WARN] No best checkpoint found. System keeps wages from last epoch.")

        model.save_val_predictions = True
        model.run_dir = str(run_dir)
        model.is_final_validation = True

        print("\n[INFO] Final validation for best model...")
        trainer.validate(model, datamodule=datamodule)

    finally:
        os.chdir(old_cwd)

    metrics_summary = {
        "experiment": exp["name"],
        "seed": exp["seed"],
        "use_water_removal": exp.get("use_water_removal", True),
        "best_mean_score": score_to_float(cb_map["mean"].best_model_score),
        "best_mean_path": str(cb_map["mean"].best_model_path),
        "best_min_score": score_to_float(cb_map["min"].best_model_score),
        "best_min_path": str(cb_map["min"].best_model_path),
        "best_shandan_v1_score": score_to_float(getattr(cb_map.get("shandan_v1"), "best_model_score", None)),
        "best_changjiang_v1_score": score_to_float(getattr(cb_map.get("changjiang_v1"), "best_model_score", None)),
        "final_model_path": str(run_dir / "full_model_final.pth"),
        "run_dir": str(run_dir),
        "train_total_uav": total_uav,
        "train_total_sat": total_sat,
        "train_total_places": total_places,
    }

    metrics_jsonl = run_dir / "val_metrics.jsonl"
    if metrics_jsonl.exists():
        with open(metrics_jsonl, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            last_entry = lines[-1]
            metrics_summary["best_epoch_used"] = last_entry.get("epoch")
            metrics_summary["best_model_metrics"] = last_entry.get("datasets", {})

    del trainer, model, datamodule
    torch.cuda.empty_cache()
    gc.collect()

    return metrics_summary


def select_interesting_queries(predictions_water: dict, predictions_nowater: dict, 
                                 num_divergent: int = 3, num_consensus: int = 2) -> tuple:
    divergent = []
    consensus_hits = []
    
    queries_w = predictions_water.get("queries", [])
    queries_n = predictions_nowater.get("queries", [])
    
    if len(queries_w) != len(queries_n):
        print(f"[WARN] Query count mismatch: water={len(queries_w)}, no-water={len(queries_n)}")
        return [], []
    
    dict_n = {q["query_path"]: q for q in queries_n}
    for qw in queries_w:
        path = qw["query_path"]
        if path not in dict_n:
            continue
        qn = dict_n[path]
        
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
        elif hit_w and hit_n:
            consensus_hits.append(entry)
            
        if len(divergent) >= num_divergent and len(consensus_hits) >= num_consensus:
            break
    
    return divergent[:num_divergent], consensus_hits[:num_consensus]

def create_comparison_figure_fixed(query_path: str, water_top1: str, nowater_top1: str,
                             output_path: Path, water_status: str, nowater_status: str, title: str = ""):

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img_q = Image.open(query_path).convert('RGB')
        axes[0].imshow(img_q)
        axes[0].set_title("Query (UAV)")
        axes[0].axis('off')
        
        if water_top1 and os.path.exists(water_top1):
            img_w = Image.open(water_top1).convert('RGB')
            axes[1].imshow(img_w)
            axes[1].set_title(f"With Water Removal\n{water_status}")
        else:
            axes[1].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[1].set_title("With Water Removal")
        axes[1].axis('off')
        
        if nowater_top1 and os.path.exists(nowater_top1):
            img_n = Image.open(nowater_top1).convert('RGB')
            axes[2].imshow(img_n)
            axes[2].set_title(f"Without Water Removal\n{nowater_status}")
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


def generate_ab_report(logs_root: Path, output_dir: Path, config: PipelineConfig, data_config: List[dict], num_examples: int = 3):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = logs_root / "ab_results.csv"
    
    if not results_csv.exists():
        print("No ab_results.csv found. Skipping report.")
        return

    df = pd.read_csv(results_csv)
    
    df['base_config'] = df['experiment'].str.replace(r'_(WATER|NOWATER)_s\d+$', '', regex=True)
    
    rows = []
    for _, row in df.iterrows():
        base = {
            "experiment": row["experiment"],
            "base_config": row["base_config"],
            "seed": row["seed"],
            "use_water_removal": row["use_water_removal"],
            "best_mean_R1": row["best_mean_score"],
            "epoch_used": row.get("best_epoch_used"),
            "train_total_uav": row.get("train_total_uav"),
            "train_total_sat": row.get("train_total_sat"),
            "train_total_places": row.get("train_total_places"),
        }
        metrics_json = row.get("best_model_metrics") or "{}"
        if isinstance(metrics_json, str):
            try:
                metrics_json = json.loads(metrics_json)
            except Exception:
                metrics_json = {}

        for ds_name, vals in metrics_json.items():
            rows.append({
                **base,
                "dataset": ds_name,
                "R1": vals.get("R1"),
                "R5": vals.get("R5"),
                "R10": vals.get("R10"),
            })
    
    report_df = pd.DataFrame(rows)
    report_path = output_dir / "ab_report_table.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Report table saved to {report_path}")

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
            
            vals_water = [water_means.get(d, np.nan) for d in datasets]
            vals_nowater = [nowater_means.get(d, np.nan) for d in datasets]
            
            ax.bar(x - width/2, vals_nowater, width, label='Baseline (WATER)', color='coral', alpha=0.8)
            ax.bar(x + width/2, vals_water, width, label='Filtered (NO_WATER)', color='seagreen', alpha=0.8)
            
            ax.set_title(f'Recall@{metric[-1]} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(0, 1.05)
        
        fig.suptitle('AB Test: Impact of Water Removal on VPR Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "ab_comparison_all_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()

    examples_dir = output_dir / "query_examples"
    examples_dir.mkdir(exist_ok=True)

    _, val_csvs_fixed = ensure_place_ids_for_variant(config, data_config, use_water_removal=True) # flaga nie ma znaczenia dla val, i tak wymusi FIXED_NO_WATER
    dm_fixed = MapsDataModule(tiles_csv_file_paths=[], batch_size=1, val_set_names=val_csvs_fixed, shuffle_all=False)
    
    ds_map_fixed = {Path(p).stem.replace("_FIXED_NO_WATER", ""): ds for p, ds in zip(dm_fixed.val_set_names, dm_fixed.val_datasets)}

    all_predictions = {}
    for _, row in df.iterrows():
        run_dir = Path(row["run_dir"])
        for pf in run_dir.glob("predictions_*.json"):
            raw_name = pf.stem.replace("predictions_", "")
            ds_name = raw_name.replace("_FIXED_NO_WATER", "") # Normalizacja nazwy
            
            key = (row["seed"], row["use_water_removal"], ds_name)
            with open(pf, 'r', encoding='utf-8') as f:
                all_predictions[key] = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example_records = []
    
    seeds = sorted(df["seed"].unique())
    base_configs = sorted(df["base_config"].unique())
    all_datasets = sorted({key[2] for key in all_predictions.keys()})
    
    for base_cfg in base_configs:
        for seed in seeds:
            for ds_name in all_datasets:
                key_water = (seed, False, ds_name) # WATER = False
                key_nowater = (seed, True, ds_name) # NOWATER = True
                
                if key_water not in all_predictions or key_nowater not in all_predictions:
                    continue
                
                preds_w = all_predictions[key_water]
                preds_n = all_predictions[key_nowater]
                
                divergent, consensus = select_interesting_queries(preds_w, preds_n, num_divergent=num_examples, num_consensus=2)
                selected_queries = divergent if divergent else consensus[:num_examples]
                
                if not selected_queries:
                    continue

                lookup_key = ds_name.replace("_FIXED_NO_WATER", "")
                ds_obj = ds_map_fixed.get(lookup_key) # Jeden wspólny dataset
                
                df_w = df[(df["base_config"] == base_cfg) & (df["seed"] == seed) & (df["use_water_removal"] == False)]
                df_n = df[(df["base_config"] == base_cfg) & (df["seed"] == seed) & (df["use_water_removal"] == True)]
                
                if df_w.empty or df_n.empty:
                    continue

                row_w = df_w.iloc[0]
                row_n = df_n.iloc[0]
                
                model_w, model_n = None, None
                try:
                    model_w = VPRModel.load_from_checkpoint(row_w["best_mean_path"], strict=True)
                except Exception as e:
                    print(f"Failed to load WATER model: {e}")
                
                try:
                    model_n = VPRModel.load_from_checkpoint(row_n["best_mean_path"], strict=True)
                except Exception as e:
                    print(f"Failed to load NOWATER model: {e}")
            
                for idx, ex in enumerate(selected_queries):
                    ex_type = "divergent" if ex in divergent else "consensus"
                    ex_dir = examples_dir / f"{base_cfg}_s{seed}_{ds_name}_{ex_type}_{idx+1:02d}"
                    ex_dir.mkdir(exist_ok=True)
                    
                    query_path = ex["query_path"]
                    
                    if model_w and ds_obj:
                        model_w.extract_attention_single_image(ds_obj, query_path, device, str(ex_dir / "water_attn"))
                    if model_n and ds_obj:
                        model_n.extract_attention_single_image(ds_obj, query_path, device, str(ex_dir / "nowater_attn"))

                    try:
                        shutil.copy2(query_path, ex_dir / "query.jpg")
                        if ex["water_top1"]: shutil.copy2(ex["water_top1"], ex_dir / "water_top1.jpg")
                        if ex["nowater_top1"]: shutil.copy2(ex["nowater_top1"], ex_dir / "nowater_top1.jpg")
                    except Exception:
                        pass
                    
                    w_status = "HIT" if ex['water_hit'] else "MISS"
                    n_status = "HIT" if ex['nowater_hit'] else "MISS"
                    
                    create_comparison_figure_fixed(
                        query_path, ex["water_top1"], ex["nowater_top1"],
                        ex_dir / "comparison.png", 
                        water_status=w_status,
                        nowater_status=n_status,
                        title=f"{base_cfg} | {ds_name} | Seed {seed}"
                    )
                    
                    example_records.append({
                        "base_config": base_cfg, "seed": seed, "dataset": ds_name, "type": ex_type,
                        "water_result": w_status, "nowater_result": n_status,
                        "example_dir": str(ex_dir.relative_to(output_dir))
                    })

                if model_w: del model_w
                if model_n: del model_n
                torch.cuda.empty_cache()

    examples_path = output_dir / "ab_query_examples.txt"
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("AB TEST: QUERY EXAMPLES & ATTENTION MAPS DIRECTORIES\n")
        f.write("=" * 80 + "\n")
        for rec in example_records:
            f.write(f"\n{rec['example_dir']}\n")
            f.write(f"  Config: {rec['base_config']} | Seed {rec['seed']} | {rec['dataset']}\n")
            f.write(f"  Water: {rec['water_result']} | No-Water: {rec['nowater_result']}\n")

    
def main():
    config = PipelineConfig()
    config.force_regenerate_place_ids = False
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

    def make_experiments_for_config(base_name, cfg, seeds):
        exps = []
        for seed in seeds:
            exps.append({**cfg, "name": f"{base_name}_WATER_s{seed}",  "seed": seed, "use_water_removal": False})
            exps.append({**cfg, "name": f"{base_name}_NOWATER_s{seed}", "seed": seed, "use_water_removal": True})
        return exps

    cfg1 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "ContrastiveLoss", "miner_name": "PairMarginMiner",
        "loss_margin": 0.8, "loss_margin_neg": 0.4,
        "miner_margin": 0.8, "miner_margin_neg": 0.6,
        "distance": "DotProductSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "multistep", "lr_sched_args": {"milestones": [20, 30], "gamma": 0.1},
    }

    cfg2 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "ContrastiveLoss", "miner_name": "MultiSimilarityMiner",
        "loss_margin": 0.8, "loss_margin_neg": 0.5,
        "miner_margin": 0.12,
        "distance": "CosineSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    cfg3 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "TripletMarginLoss", "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05, "miner_margin": 0.05,
        "type_of_triplets": "all", "swap": False, "smooth_loss": False,
        "distance": "CosineSimilarity",
        "agg_arch": "GeM", "agg_config": {"p": 3, "eps": 1e-6},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    cfg4 = {
        "max_epochs": 40, "batch_size": 32,
        "loss_name": "TripletMarginLoss", "miner_name": "TripletMarginMiner",
        "loss_margin": 0.05, "miner_margin": 0.05,
        "type_of_triplets": "all", "swap": False, "smooth_loss": False,
        "distance": "CosineSimilarity",
        "agg_arch": "ConvAP", "agg_config": {"in_channels": 2048, "out_channels": 512, "s1": 2, "s2": 2},
        "lr_sched": "cosine", "lr_sched_args": {"T_max": 35},
    }

    seeds = [42, 123, 999]
    EXPERIMENTS = []
    EXPERIMENTS += make_experiments_for_config("C1_ContrastivePair_GeM", cfg1, seeds)
    EXPERIMENTS += make_experiments_for_config("C2_ContrastiveMS_GeM", cfg2, seeds)
    EXPERIMENTS += make_experiments_for_config("C3_TripletTriplet_GeM", cfg3, seeds)
    EXPERIMENTS += make_experiments_for_config("C4_TripletTriplet_ConvAP", cfg4, seeds)

    RUN_VARIANT = "both"
    if RUN_VARIANT == "water_only":
        EXPERIMENTS = [e for e in EXPERIMENTS if e.get("use_water_removal", False)]
    elif RUN_VARIANT == "nowater_only":
        EXPERIMENTS = [e for e in EXPERIMENTS if not e.get("use_water_removal", True)]
    
    logs_root = Path("./logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    
    ab_results_path = logs_root / "ab_results.csv"
    all_results = []
    
    if ab_results_path.exists():
        try:
            existing_df = pd.read_csv(ab_results_path)
            all_results = existing_df.to_dict('records')
            print(f"[RESUME] Loaded {len(all_results)} existing results from {ab_results_path.name}")
        except Exception as e:
            print(f"[WARN] Could not load existing results: {e}")
            all_results = []
    
    SKIP_EXISTING_RUNS = True
    if SKIP_EXISTING_RUNS and all_results:
        done_names = {str(r.get('experiment', '')) for r in all_results}
        original_count = len(EXPERIMENTS)
        EXPERIMENTS = [e for e in EXPERIMENTS if e['name'] not in done_names]
        skipped = original_count - len(EXPERIMENTS)
        if skipped > 0:
            print(f"[SKIP] {skipped} runs already done, {len(EXPERIMENTS)} remaining")

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
        )
        all_results.append(result)
        
        pd.DataFrame(all_results).to_csv(ab_results_path, index=False)
        remaining = len(EXPERIMENTS) - EXPERIMENTS.index(exp) - 1
        print(f"\n[SAVE] Progress: {len(all_results)}/{len(all_results) + remaining} saved to {ab_results_path}")

    print("\n" + "=" * 100)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 100)

    generate_ab_report(
        logs_root=logs_root, 
        output_dir=logs_root / "report",
        config=config,
        data_config=DATA_CONFIG,
        num_examples=3
    )


if __name__ == "__main__":
    main()