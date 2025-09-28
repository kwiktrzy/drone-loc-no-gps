import pytorch_lightning as pl

from dataloaders.MapsDataloader import MapsDataModule
from dataset_splitter.MapSatellite import MapSatellite
from dataset_splitter.ThumbnailsGenerator import ThumbnailsGenerator
from dataset_splitter.UavCropGenerator import UavCropGenerator
from dataset_splitter.PlaceIdGenerator import PlaceIdGenerator
from vpr_model import VPRModel
import pandas as pd

if __name__ == '__main__':
    visloc_satelite_taizhou_output_csv='/workspace/repos/drone-loc-no-gps/Dataframes/Taizhou-1.csv'
    aerialvl_satelite_shandong_output_csv='/workspace/repos/drone-loc-no-gps/Dataframes/Shandong-1.csv'

    thumbnails_generator = ThumbnailsGenerator(
        output_dir='/workspace/repos/datasets/train_thumbnails',
        satellite_map_names=[
            MapSatellite(csv_path='/workspace/repos/datasets/UAV_VisLoc_dataset/satellite_ coordinates_range.csv',
                         thumbnails_satellite_csv_output_path=visloc_satelite_taizhou_output_csv,
                         map_tif_path='/workspace/repos/datasets/UAV_VisLoc_dataset/03/satellite03.tif',
                         map_name='satellite03.tif',
                         region_name='Taizhou-1',
                         friendly_name='visloc-Taizhou-1-03'),
            MapSatellite(csv_path='/workspace/repos/datasets/Aerial_VL_dataset/coordinates_range.csv',
                         thumbnails_satellite_csv_output_path=aerialvl_satelite_shandong_output_csv,
                         map_tif_path='/workspace/repos/datasets/Aerial_VL_dataset/geo_referenced_map/@small_map@120.42114259488751@36.604504047017464@120.4568481612987@36.586863027841225@.tif',
                         map_name='@small_map@120.42114259488751@36.604504047017464@120.4568481612987@36.586863027841225@.tif',
                         region_name='Shandong-1',
                         friendly_name='aerialal-Shandong-1-01')            
        ],
        is_rebuild_csv=False,
        height_size=224,
        width_size=224
    )

    thumbnails_generator.generate_thumbnails()

    uav_visloc = UavCropGenerator(
        csv_path='/workspace/repos/datasets/UAV_VisLoc_dataset/03/03.csv',
        cropped_uav_csv_output_path=visloc_satelite_taizhou_output_csv,
        cropped_output_dir='/workspace/repos/datasets/train_thumbnails',
        uav_images_dir='/workspace/repos/datasets/UAV_VisLoc_dataset/03/drone',
        region_name='Taizhou-1',
        friendly_name='visloc-Taizhou-1-03-uav'
    )
    uav_visloc.generate_thumbnails()
    # TODO: smart if

    place_id_generator = PlaceIdGenerator(
        csv_thumbnails_paths=[visloc_satelite_taizhou_output_csv]
    )

    datamodule = MapsDataModule(
        thumbnails_csv_file_paths=[visloc_satelite_taizhou_output_csv],
        batch_size=32,
        val_set_dataframes_paths=[aerialvl_satelite_shandong_output_csv]
    )

    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )


    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='Shandong-1_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{Shandong-1_val/R1:.4f}]_R5[{Shandong-1_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=4,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)