import json

import torch
import wandb
from egg.core import Interaction
from egg.core.callbacks import WandbLogger
from egg.core.language_analysis import TopographicSimilarity


class CustomTopSimWithWandbLogging(TopographicSimilarity):
    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = [msg.tolist() for msg in messages]
        sender_input = torch.flatten(logs.sender_input, start_dim=1)

        # topsim = self.compute_topsim(sender_input, messages, self.sender_input_distance_fn, self.message_distance_fn)
        topsim = self.compute_topsim(
            sender_input,
            logs.aux_input["scores"],
            self.sender_input_distance_fn,
            self.message_distance_fn,
        )

        logs.aux["topsim"] = torch.as_tensor(topsim)

        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))
        print(output, flush=True)

        wandb_dict = {f"{mode}/topsim": topsim, "Epoch": epoch}
        wandb.log(wandb_dict)


class CustomWandbLogger(WandbLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.img_msg_corr = {}

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        # accuracy, loss, cosine similarity, messages
        if logs == Interaction.empty():
            return

        acc = logs.aux["acc"].mean()
        loss = loss.detach()

        batch_dict = dict(batch_loss=loss, batch_acc=acc)

        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb(batch_dict, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.general_end_of_epoch(logs, loss, epoch, "train")

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):

        self.general_end_of_epoch(logs, loss, epoch, "valid")

    def general_end_of_epoch(
        self, logs: Interaction, loss: float, epoch: int, split: str
    ):
        if logs == Interaction.empty():
            return

        acc = logs.aux["acc"].mean()
        img_ids = logs.aux.pop("img_id")

        for idx in range(len(img_ids)):
            img_id = img_ids[idx].item()
            msg = logs.message[idx]
            try:
                # try/except faster in further epochs
                self.img_msg_corr[img_id].append(msg)
            except KeyError:
                self.img_msg_corr[img_id] = [msg]

        log_dict = {
            f"{split}_loss": loss,
            f"{split}_acc": acc,
            "epoch": epoch,
        }

        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(log_dict, commit=True)
