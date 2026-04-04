from typing import List
import gc
import json
import os
import shutil
import inspect
import copy
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.place_id_generators.ManyToManyPlaceIdGenerator import ManyToManyPlaceIdGenerator
from dataset_splitter.structs.MapSatellite import MapSatellite
from dataset_splitter.satellite_generators.OverlapingTilesGenerator import OverlapingTilesGenerator
from dataset_splitter.uav_generators.UavSmallerCropGenerator import UavSmallerCropGenerator
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


def main():
    config = PipelineConfig()
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
    all_csv_paths_overlapping_patches = {}

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
                # --- Satellite Tile Generation ---
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

                # --- UAV Crop Generation ---
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

    # --- Place ID Generation ---
    print("\n--- Starting Place ID Generation ---")
    for d_conf in DATA_CONFIG:
        region_name = d_conf["region_name"]
        csv_path = all_csv_paths_one_to_one[region_name]
        is_val = d_conf["set_type"] == "val"

        csv_output_path = get_processed_path(csv_path, d_conf["output_suffix"])
        generator = ManyToManyPlaceIdGenerator(
            csv_tiles_path=csv_path,
            csv_place_ids_output_path=csv_output_path,
            force_regenerate=config.force_regenerate_place_ids,
            is_validation_set=is_val,
            is_validation_set_v2=d_conf.get("val_variant") == "v2",
            radius_neighbors_meters=70 if is_val else d_conf["crop_range_meters"],
            tiles_trash_directory=config.DATAFRAMES_TILES_TRASH,
        )

        generator.generate_place_ids()

    # --- Prepare Data for Model Training ---
    train_csvs: List[str] = []
    val_csvs: List[str] = []

    for d in DATA_CONFIG:
        base_path = str(config.DATAFRAMES_ONE_TO_ONE_DIR / f"{d['region_name']}.csv")
        final_path = get_processed_path(base_path, d["output_suffix"])

        if d["set_type"] == "train":
            train_csvs.append(final_path)
        elif d["set_type"] == "val":
            val_csvs.append(final_path)

    print(f"Train CSVs count: {len(train_csvs)}")
    print(f"Val CSVs count: {len(val_csvs)}")

    EXPERIMENTS = [
        {
            "name": "EXP-054_GeM_tripletall_bs32_cyclic_triangular2",
            "seed": 42,
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
                "warmup_fraction": 0.05,   # 5% total steps = ~2-3 epoki
                "eta_min_ratio": 0.01,     # eta_min = 0.01 * 1e-4 = 1e-6
            },
        },
        # {
        #     "name": "EXP-055_GeM_tripletall_bs32_cyclic_expRange",
        #     "seed": 42,
        #     "max_epochs": 40,
        #     "loss_name": "TripletMarginLoss",
        #     "miner_name": "TripletMarginMiner",
        #     "loss_margin": 0.05,
        #     "miner_margin": 0.05,
        #     "type_of_triplets": "all",
        #     "swap": False,
        #     "smooth_loss": False,
        #     "agg_arch": "GeM",
        #     "agg_config": {"p": 3, "eps": 1e-6},
        #     "lr_sched": "cyclic",
        #     "lr_sched_args": {
        #         "base_lr": 2e-5,
        #         "max_lr": 1e-4,
        #         "mode": "exp_range",
        #         "gamma": 0.9995,
        #         "step_size_up_epochs": 2,
        #         "step_size_down_epochs": 2
        #     }
        # },
    ]

    logs_root = Path("./logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 100)
        print(f"STARTING EXPERIMENT: {exp['name']}")
        print("=" * 100)

        pl.seed_everything(exp["seed"], workers=True)

        run_dir = (logs_root / exp["name"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "experiment_config.json", "w") as f:
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

        finally:
            os.chdir(old_cwd)

        summary_row = copy.deepcopy(exp)
        
        current_agg_arch = exp.get("agg_arch", "gem")
        current_agg_config = exp.get("agg_config", {})
        if current_agg_arch == "ConvAP":
            summary_row["convap_out_channels"] = current_agg_config.get("out_channels")
            summary_row["convap_s1"] = current_agg_config.get("s1")
            summary_row["convap_s2"] = current_agg_config.get("s2")

        summary_row.update({
            "best_mean_score": score_to_float(cb_map["mean"].best_model_score),
            "best_mean_path": cb_map["mean"].best_model_path,
            "best_min_score": score_to_float(cb_map["min"].best_model_score),
            "best_min_path": cb_map["min"].best_model_path,
            "best_shandan_v1_score": score_to_float(cb_map["shandan_v1"].best_model_score),
            "best_shandan_v1_path": cb_map["shandan_v1"].best_model_path,
            "best_changjiang_v1_score": score_to_float(cb_map["changjiang_v1"].best_model_score),
            "best_changjiang_v1_path": cb_map["changjiang_v1"].best_model_path,
            "final_model_path": str(run_dir / "full_model_final.pth"),
        })

        all_summary_rows.append(summary_row)
        
        pd.DataFrame(all_summary_rows).to_csv(logs_root / "screening_summary.csv", index=False)

        print("FINISHED EXPERIMENT:")
        print(summary_row)

        del trainer
        del model
        del datamodule
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll experiments finished.")
    print(f"Summary saved to: {logs_root / 'screening_summary.csv'}")


if __name__ == "__main__":
    main()