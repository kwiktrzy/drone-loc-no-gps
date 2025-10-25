import pytorch_lightning as pl
from vpr_model import VPRModel
from dataloaders.MapsDataloader import MapsDataModule

checkpoint_path = "/workspace/repos/logs/lightning_logs/version_9/checkpoints/last.ckpt"

# model = VPRModel.load_from_checkpoint(checkpoint_path)

model = VPRModel.load_from_checkpoint(checkpoint_path, strict=False)

visloc_satelite_taizhou_output_csv = (
    "/workspace/repos/drone-loc-no-gps/Dataframes/Taizhou-1.csv"
)
aerialvl_satelite_shandong_output_csv = (
    "/workspace/repos/drone-loc-no-gps/Dataframes/Shandong-1.csv"
)

visloc_satelite_shandan_output_csv = (
    "/workspace/repos/drone-loc-no-gps/Dataframes/Shandan.csv"
)
visloc_satelite_shandan_output_csv = (
    "/workspace/repos/drone-loc-no-gps/Dataframes/Shandan.csv"
)
visloc_satelite_yunnan_output_csv = (
    "/workspace/repos/drone-loc-no-gps/Dataframes/Yunnan.csv"
)

datamodule = MapsDataModule(
    tiles_csv_file_paths=[visloc_satelite_taizhou_output_csv],
    batch_size=32,
    val_set_names=[visloc_satelite_shandan_output_csv],
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
)


metrics = trainer.validate(model=model, datamodule=datamodule)
print(metrics)
print("Done.")
