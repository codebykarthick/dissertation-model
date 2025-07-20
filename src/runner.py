import os
import sys

import torch.multiprocessing as mp

from util.arg_checks import create_parser, validate_args
from util.logger import setup_logger
from util.trainers.classification_trainer import (
    ClassificationCrossValidationTrainer,
    ClassificationTrainer,
)
from util.trainers.distill_trainer import DistillationTrainer
from util.trainers.export_trainer import ExportTrainer
from util.trainers.grad_cam import GradCamBench
from util.trainers.mc_dropout_trainer import MCDropoutTrainer
from util.trainers.siamese_trainer import SiameseTrainer

log = setup_logger()


class Runner:
    def __init__(self, model_name: str, lr: float, epochs: int,
                 is_loss_weighted: bool, is_oversampled: bool,
                 batch_size: int, patience: int,
                 roi: bool, roi_weight: str, fill_noise: bool, num_workers: int, task_type: str,
                 label: str, k: int = 10, filename: str = "", temperature: float = 2.0,
                 teacher1: str = "", teacher2: str = "", alpha: float = 0.5):
        if task_type == "classification_crossval":
            self.trainer = ClassificationCrossValidationTrainer(
                k=k, fill_noise=fill_noise, model_name=model_name, lr=lr,
                epochs=epochs, is_loss_weighted=is_loss_weighted, is_sampling_weighted=is_oversampled,
                batch_size=batch_size, patience=patience, roi=roi, roi_weight=roi_weight,
                num_workers=num_workers, task_type=task_type, label=label)
        elif task_type == "classification":
            self.trainer = ClassificationTrainer(
                k=k, fill_noise=fill_noise, model_name=model_name, lr=lr,
                epochs=epochs, is_loss_weighted=is_loss_weighted, is_sampling_weighted=is_oversampled,
                batch_size=batch_size, patience=patience, roi=roi, roi_weight=roi_weight,
                num_workers=num_workers, task_type=task_type, label=label, filename=filename)
        elif task_type == "siamese":
            self.trainer = SiameseTrainer(
                roi=roi, fill_noise=fill_noise, model_name=model_name, roi_weight=roi_weight,
                num_workers=num_workers, k=k, is_sampling_weighted=is_oversampled,
                is_loss_weighted=is_loss_weighted, batch_size=batch_size, epochs=epochs, lr=lr,
                task_type=task_type, patience=patience, label=label, filename=filename)
        elif task_type == "distillation":
            self.trainer = DistillationTrainer(
                roi=roi, fill_noise=fill_noise, model_name=model_name, roi_weight=roi_weight,
                num_workers=num_workers, k=k, is_sampling_weighted=is_oversampled,
                is_loss_weighted=is_loss_weighted, batch_size=batch_size, epochs=epochs, lr=lr,
                task_type=task_type, patience=patience, label=label, filename=filename,
                temperature=temperature, teacher1=teacher1, teacher2=teacher2, alpha=alpha)
        elif task_type == "gradcam":
            self.trainer = GradCamBench(k=k, fill_noise=fill_noise, model_name=model_name, lr=lr,
                                        epochs=epochs, is_loss_weighted=is_loss_weighted, is_sampling_weighted=is_oversampled,
                                        batch_size=batch_size, patience=patience, roi=roi, roi_weight=roi_weight,
                                        num_workers=num_workers, task_type=task_type, label=label, filename=filename)
        elif task_type == "mcdropout":
            self.trainer = MCDropoutTrainer(k=k, fill_noise=fill_noise, model_name=model_name, lr=lr,
                                            epochs=epochs, is_loss_weighted=is_loss_weighted, is_sampling_weighted=is_oversampled,
                                            batch_size=batch_size, patience=patience, roi=roi, roi_weight=roi_weight,
                                            num_workers=num_workers, task_type=task_type, label=label, filename=filename)
        elif task_type == "mobile":
            self.trainer = ExportTrainer(
                model_name=model_name, task_type=task_type, model_filepath=filename, script_modelpath=filename)

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()

    def export(self):
        self.trainer.export()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    valid_models = {
        "classification": ["mobilenetv3", "cnn", "efficientnet", "shufflenet", "tinyvit"],
        "classification_crossval": ["mobilenetv3", "cnn", "efficientnet", "shufflenet", "tinyvit"],
        "siamese": ["mobilenetv3", "efficientnet", "shufflenet"],
        "distillation": "student",
        "gradcam": ["student", "efficientnet", "shufflenet"],
        "mcdropout": ["efficientnet", "shufflenet"]
    }

    parser = create_parser()
    args = parser.parse_args()

    # Manual validation of certain argument values
    validate_args(parser, valid_models, args)

    # Unpack all the arguments
    list_of_models = args.models
    mode = args.mode
    lr = args.lr
    epochs = args.epochs
    weighted_loss = args.weighted_loss
    weighted_sampling = args.weighted_sampling
    file_name = args.file if args.file else ""
    batch_size = args.batch
    patience = args.patience
    roi = args.roi
    roi_weight = args.roi_weight
    fill_noise = args.fill_noise
    num_workers = args.workers
    k = args.k_fold
    task_type = args.task_type
    label = args.label
    teacher1 = args.teacher1
    teacher2 = args.teacher2
    temperature = args.temperature
    alpha = args.alpha

    for model_name in list_of_models:
        runner = Runner(lr=lr, epochs=epochs, is_loss_weighted=weighted_loss,
                        is_oversampled=weighted_sampling, batch_size=batch_size, patience=patience,
                        model_name=model_name, roi=roi, roi_weight=roi_weight, fill_noise=fill_noise, num_workers=num_workers,
                        task_type=task_type, label=label, k=k, filename=file_name, teacher1=teacher1, teacher2=teacher2,
                        temperature=temperature, alpha=alpha)

        if mode == "train":
            if not os.path.exists("dataset/Images"):
                log.error(
                    "Dataset does not exist for training, please download using data_setup.py before training.")
                sys.exit(1)
            runner.train()
        if mode == "evaluate":
            runner.evaluate()
        elif mode == "export":
            runner.export()
