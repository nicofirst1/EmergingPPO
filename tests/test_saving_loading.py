import os
from torch import nn
import wandb
from src.saver import ModelSaverCallback




def test_save_model():


    wandb.init(project="test_project", mode="offline")
    # define empty nn.Model
    model = nn.Module()


    # define the path to save the model
    save_path = ".logs/"

    saver= ModelSaverCallback(model, save_path, "test", save_every_n_epochs=1)


    saver.on_epoch_end(0, 0, 0)

    # check if the model is saved
    assert os.path.exists(".logs/test_model_0.pth")

    # remove the saved model
    os.remove(".logs/test_model_0.pth")

    print("Test passed")


if __name__ == "__main__":
    test_save_model()
    