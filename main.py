from turtle import st
from typing import List
import pytorch_lightning as pl

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.MapSatellite import MapSatellite
from dataset_splitter.TilesGenerator import TilesGenerator
from dataset_splitter.OverlapingTilesGenerator import OverlapingTilesGenerator
from dataset_splitter.UavCropGenerator import UavCropGenerator
from dataset_splitter.PlaceIdGenerator import PlaceIdGenerator
from vpr_model import VPRModel
import pandas as pd
from pathlib import Path
import os
import torch
import shutil


class PipelineConfig:
    def __init__(self, project_root="/workspace"):
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

    if not (output_csv_path.exists() and thumb_dir.exists() and any(thumb_dir.iterdir())):
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


def main():
    config = PipelineConfig()
    config.DATAFRAMES_ROOT.mkdir(parents=True, exist_ok=True)

    DATA_CONFIG = [
        {
            "set_type": "train",
            "region_name": "Taizhou-1",
            "uav_visloc_id": "03",
            "map_filename": "satellite03.tif",
        },
        {
            "set_type": "train",
            "region_name": "Yunnan",
            "uav_visloc_id": "05",
            "map_filename": "satellite05.tif",
        },
        {
            "set_type": "train",
            "region_name": "Changjiang-20",
            "uav_visloc_id": "01",
            "map_filename": "satellite01.tif",
        },
        {
            "set_type": "train",
            "region_name": "Changjiang-23",
            "uav_visloc_id": "02",
            "map_filename": "satellite02.tif",
        },
        {
            "set_type": "train",
            "region_name": "Taizhou-6",
            "uav_visloc_id": "04",
            "map_filename": "satellite04.tif",
        },
        {
            "set_type": "train",
            "region_name": "Zhuxi",
            "uav_visloc_id": "06",
            "map_filename": "satellite06.tif",
        },
        {
            "set_type": "train",
            "region_name": "Donghuayuan",
            "uav_visloc_id": "07",
            "map_filename": "satellite07.tif",
        },
        {
            "set_type": "train",
            "region_name": "Huzhou-3",
            "uav_visloc_id": "08",
            "map_filename": "satellite08.tif",
        },
        # {
        #     "set_type": "train",
        #     "region_name": "Huzhou-3-1",
        #     "uav_visloc_id": "09",
        #     "map_filename": "satellite09.tif",
        # },
        {
            "set_type": "train",
            "region_name": "Huailai",
            "uav_visloc_id": "10",
            "map_filename": "satellite10.tif",
        },
        {
            "set_type": "val",
            "region_name": "Shandan",
            "uav_visloc_id": "11",
            "map_filename": "satellite11.tif",
        },
        # {'set_type': 'Huailai', 'region_name': 'Shandong-1', 'dataset_type': 'AerialVL'},
    ]

    all_csv_paths_one_to_one = {}
    all_csv_paths_overlapping_patches = {}

    for d_conf in DATA_CONFIG:
        region_name = d_conf["region_name"]

        if config.one_to_one_tiles:
            output_csv_path = config.DATAFRAMES_ONE_TO_ONE_DIR / f"{region_name}.csv"
            all_csv_paths_one_to_one[region_name] = str(output_csv_path)
            thumb_dir = config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR / region_name

            skip_generation = clearup_generated_data(config, output_csv_path, thumb_dir, region_name)
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
                thumb_gen = TilesGenerator(
                    output_dir=str(config.THUMBNAILS_ONE_TO_ONE_OUTPUT_DIR),
                    satellite_map_names=[map_sat],
                    is_rebuild_csv=config.force_regenerate_tiles,  # Rebuild for each new region processing
                )
                thumb_gen.generate_tiles()

                # --- UAV Crop Generation ---
                uav_gen = UavCropGenerator(
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
        if config.overlapping_patches_tiles:
            output_csv_path = config.DATAFRAMES_OVERLAPPING_PATCHES_DIR / f"{region_name}.csv"
            all_csv_paths_overlapping_patches[region_name] = str(output_csv_path)
            thumb_dir = config.THUMBNAILS_OVERLAPPING_PATCHES_OUTPUT_DIR / region_name

            skip_generation = clearup_generated_data(config, output_csv_path, thumb_dir, region_name)
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
                    output_dir=str(config.THUMBNAILS_OVERLAPPING_PATCHES_OUTPUT_DIR),
                    satellite_map_names=[map_sat],
                    is_rebuild_csv=True,  # Rebuild for each new region processing
                )
                thumb_gen.generate_tiles()
                # --- UAV Crop Generation ---
                uav_gen = UavCropGenerator(
                    csv_path=str(
                        config.UAV_VISLOC_ROOT
                        / d_conf["uav_visloc_id"]
                        / f"{d_conf['uav_visloc_id']}.csv"
                    ),
                    cropped_uav_csv_output_path=str(output_csv_path),
                    cropped_output_dir=str(
                        config.THUMBNAILS_OVERLAPPING_PATCHES_OUTPUT_DIR
                    ),
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
        if config.one_to_one_tiles:
            csv_path = all_csv_paths_one_to_one[region_name]
            is_val = d_conf["set_type"] == "val"
            PlaceIdGenerator(
                csv_tiles_paths=[csv_path],
                is_validation_set=is_val,
                force_regenerate=config.force_regenerate_place_ids,
            )

    # --- Prepare Data for Model Training ---
    train_csvs: List[str] = [
        path
        for name, path in all_csv_paths_one_to_one.items()
        if any(
            d["region_name"] == name and d["set_type"] == "train" for d in DATA_CONFIG
        )
    ]
    val_csvs: List[str] = [
        path
        for name, path in all_csv_paths_one_to_one.items()
        if any(d["region_name"] == name and d["set_type"] == "val" for d in DATA_CONFIG)
    ]

    # Keep original variables for datamodule and checkpoint as requested
    visloc_satelite_taizhou_output_csv = all_csv_paths_one_to_one.get("Taizhou-1")
    visloc_satelite_yunnan_output_csv = all_csv_paths_one_to_one.get("Yunnan")
    visloc_satelite_Donghuayuan_output_csv = all_csv_paths_one_to_one.get("Donghuayuan")
    visloc_satelite_Zhuxi_output_csv = all_csv_paths_one_to_one.get("Zhuxi")
    visloc_satelite_Taizhou_6_output_csv = all_csv_paths_one_to_one.get("Taizhou-6")
    visloc_satelite_Changjiang_23_output_csv = all_csv_paths_one_to_one.get(
        "Changjiang-23"
    )
    visloc_satelite_Changjiang_20_output_csv = all_csv_paths_one_to_one.get(
        "Changjiang-20"
    )
    visloc_satelite_shandan_output_csv = all_csv_paths_one_to_one.get("Shandan")

    datamodule = MapsDataModule(  # As requested, this part is kept as is
        tiles_csv_file_paths=train_csvs, batch_size=32, val_set_names=val_csvs
    )

    model = VPRModel(
        # ---- Encoder
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },
        lr=6e-5,
        optimizer="adamw",
        weight_decay=9.5e-9,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },
        # ----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val

    # TODO thats bad model check set, refactor it
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="Shandan/R1",
        filename=f"{model.encoder_arch}"
        + "_({epoch:02d})_R1[{Shandan/R1:.4f}]_R5[{Shandan/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=f"./logs/",  # Tensorflow can be used to viz
        num_nodes=1,
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision="16-mixed",  # we use half precision to reduce  memory usage
        max_epochs=80,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[
            checkpoint_cb
        ],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)

    FULL_MODEL_PATH = "full_model.pth"
    torch.save(model.state_dict(), FULL_MODEL_PATH)
    print(f"Saved model state_dict to: {FULL_MODEL_PATH}")


if __name__ == "__main__":
    main()
