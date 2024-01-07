# -*- coding: utf-8 -*-
from roboflow import Roboflow
rf = Roboflow(api_key="API")
project = rf.workspace("msa-b0qan").project("solar-cell-6pvhl")
dataset = project.version(4).download("folder")

import cv2
from pathlib import Path
import shutil

shutil.move("solar-cell-4/train/Crack", "solar-cell-4/train/abnormal")
shutil.move("solar-cell-4/train/Not_Crack", "solar-cell-4/train/normal")

"""# Train"""

import anomalib

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import get_experiment_logger

from pytorch_lightning import Trainer, seed_everything

import yaml
from pathlib import Path
from ast import literal_eval

model = "fastflow"
image_folder = "solar-cell-4/train"
batch_size = 32
val_ratio = 0.1

# model_config = yaml.safe_load(open(f"./src/anomalib/src/anomalib/models/{model}/config.yaml", "r"))
# config = model_config
config_path = (
    Path(f"{anomalib.__file__}").parent / f"models/{model}/config.yaml"
)
config = get_configurable_parameters(model_name=model, config_path=config_path)
config["dataset"] = yaml.safe_load(open("./config.yaml", "r"))
config["trainer"].update({"default_root_dir":"results/custom/run",
                          "max_epochs": 12})
config["project"].update({"path":"results/custom/run"})
config["optimization"].update({"export_mode":"torch"})

# del config["early_stopping"]

data_config = {
    "format": "folder",
    "name": str(Path(image_folder).name),
    "root": str(Path(image_folder)),
    "path": str(Path(image_folder)),
    "val_split_ratio": float(val_ratio),
    "train_batch_size": int(batch_size),
    "test_batch_size": int(batch_size),
}

config["dataset"].update(data_config)

if config.project.get("seed") is not None:
    seed_everything(config.project.seed)

yaml.dump(literal_eval(str(config)), open("config_dump.yaml","w"))

datamodule = get_datamodule(config)
model = get_model(config)
experiment_logger = get_experiment_logger(config)
callbacks = get_callbacks(config)

trainer = Trainer(
    **config.trainer, logger=experiment_logger, callbacks=callbacks
)

trainer.fit(model=model, datamodule=datamodule)

load_model_callback = LoadModelCallback(
    weights_path=trainer.checkpoint_callback.best_model_path
)
trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

"""# Inference"""

from anomalib.deploy import TorchInferencer
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("solar-cell-4/train/abnormal/cell0012_png.rf.b89214d7061900c79a2b0d8007c5122e.jpg")[...,::-1]

inferencer = TorchInferencer(path="results/custom/run/weights/torch/model.pt")

predictions = inferencer.predict(image=image)

fig, axes = plt.subplots(1, 4)
fig.set_size_inches(10, 3)
axes[0].imshow(predictions.image)
axes[1].imshow(predictions.anomaly_map)
axes[2].imshow(predictions.pred_mask)
axes[3].imshow(predictions.segmentations)