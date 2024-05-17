import os

import torch
import wandb
from egg.core.callbacks import Callback


class ModelSaverCallback(Callback):
    def __init__(self, model, save_path, model_name, save_every_n_epochs=50):
        self.model = model
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs
        self.model_name = model_name

        # Create the save directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, loss, logs, epoch: int):
        if epoch % self.save_every_n_epochs == 0:
            self.save_model(epoch)

    def save_model(self, epochs):
        save_file = os.path.join(
            self.save_path, f"{self.model_name}_model_{epochs}.pth"
        )
        torch.save(self.model.state_dict(), save_file)
        wandb.save(save_file)
        print(f"Model saved to {save_file} at batch {epochs}")
