from trainer import Trainer


class SiameseTrainer(Trainer):
    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, roi_weight: str = ""):
        super().__init__(roi, fill_noise, model_name, num_workers, k, is_sampling_weighted,
                         is_loss_weighted, batch_size, epochs, task_type, lr, patience, roi_weight)

    def train(self):
        return super().train()

    def evaluate(self):
        ...
