from egg.core import Interaction
from egg.core.callbacks import WandbLogger


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
