from turtle import st
from typing import List
import pytorch_lightning as pl

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.place_id_generators.ManyToManyPlaceIdGenerator import ManyToManyPlaceIdGenerator
from dataset_splitter.structs.MapSatellite import MapSatellite
from dataset_splitter.satellite_generators.TilesGenerator import TilesGenerator
from dataset_splitter.satellite_generators.OverlapingTilesGenerator import OverlapingTilesGenerator
from dataset_splitter.uav_generators.UavSmallerCropGenerator import UavSmallerCropGenerator
from dataset_splitter.uav_generators.UavCropGenerator import UavCropGenerator
from dataset_splitter.place_id_generators.PlaceIdGenerator import PlaceIdGenerator
from vpr_model import VPRModel
import pandas as pd
from pathlib import Path
import os
import torch
import shutil


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
        # # {
        # #     "set_type": "train",
        # #     "region_name": "Donghuayuan",
        # #     "uav_visloc_id": "07",
        # #     "map_filename": "satellite07.tif",
        # #     "output_suffix": "one_to_one"            
        # # },
        {
            "set_type": "train",
            "region_name": "Huzhou-3",
            "uav_visloc_id": "08",
            "map_filename": "satellite08.tif",
            "crop_range_meters": 320,
            "overlap_stride_meters": 220,
            "output_suffix": "one_to_one"            
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
                    crop_range_meters=d_conf['crop_range_meters'],
                    overlap_stride_meters=d_conf['overlap_stride_meters'],
                    is_rebuild_csv=config.force_regenerate_tiles,  # Rebuild for each new region processing
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
        #todo if set is val then calculate place id for sat first.  
        is_val = d_conf["set_type"] == "val"
        
        csv_output_path = get_processed_path(csv_path, d_conf["output_suffix"])
        generator = ManyToManyPlaceIdGenerator(
                csv_tiles_path=csv_path,
                csv_place_ids_output_path=csv_output_path,
                force_regenerate=config.force_regenerate_place_ids,
                is_validation_set=is_val,
                is_validation_set_v2=d_conf.get("val_variant") == "v2",
                # radius_neighbors_meters=70,
                radius_neighbors_meters=70 if is_val else d_conf['crop_range_meters'],
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

    datamodule = MapsDataModule(  # As requested, this part is kept as is
        tiles_csv_file_paths=train_csvs, batch_size=32, val_set_names=val_csvs
    )

    model = VPRModel(
        # ---- Encoder
        backbone_arch="resnet50",
        backbone_config={
            "pretrained": True,
            "layers_to_freeze": 2,
        },
        agg_arch="gem",
        agg_config={
            "p": 3,
            "eps": 1e-6
        },
        lr=1e-4,
        optimizer="adamw",
        weight_decay=1e-4,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched="multistep",
        lr_sched_args={
            "milestones": [40, 65],
            "gamma": 0.1
        },
        # ----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name="TripletMarginLoss",
        miner_name="TripletMarginMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.2,
        faiss_gpu=False,
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val

    # TODO thats bad model check set, refactor it
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="Shandan-v1_one_to_one/R1",
        filename=f"{model.encoder_arch}" + "_v1_({epoch:02d})_R1[{Shandan-v1_one_to_one/R1:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    # TODO thats bad model check set, refactor it
    checkpoint_cb_v2 = pl.callbacks.ModelCheckpoint(
        monitor="Shandan-v2_one_to_one/R1",
        filename=f"{model.encoder_arch}" + "_v2_({epoch:02d})_R1[{Shandan-v2_one_to_one/R1:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=False,
        mode="max",
    )
    checkpoint_cb_ch23 = pl.callbacks.ModelCheckpoint(
        monitor="Changjiang-23-v1_one_to_one/R1",
        filename=f"{model.encoder_arch}" + "_v1_({epoch:02d})_R1[{Changjiang-23-v1_one_to_one/R1:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=False,
        mode="max",
    )
    checkpoint_cb_v2_ch23 = pl.callbacks.ModelCheckpoint(
        monitor="Changjiang-23-v2_one_to_one/R1",
        filename=f"{model.encoder_arch}" + "_v2_({epoch:02d})_R1[{Changjiang-23-v2_one_to_one/R1:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=False,
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
            checkpoint_cb,
            checkpoint_cb_v2,
            checkpoint_cb_ch23,
            checkpoint_cb_v2_ch23
        ],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)

    FULL_MODEL_PATH = "full_model.pth"
    torch.save(model.state_dict(), FULL_MODEL_PATH)
    print(f"Saved model state_dict to: {FULL_MODEL_PATH}")


if __name__ == "__main__":
    main()
