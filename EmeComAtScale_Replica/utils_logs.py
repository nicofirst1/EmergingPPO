from egg.core import Interaction
from egg.core.callbacks import WandbLogger


class CustomWandbLogger(WandbLogger):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_batch_end(
            self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        # accuracy, loss, cosine similarity, messages

        acc = logs.aux['acc'].mean()
        loss = loss.detach()

        batch_dict = dict(
            batch_loss=loss,
            batch_acc=acc
        )

        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb(batch_dict, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):

        acc = logs.aux['acc'].mean()
        # loss = loss.detach()

        log_dict = dict(
            train_loss=loss,
            train_acc=acc,
            epoch=epoch,
        )

        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(log_dict, commit=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        acc = logs.aux['acc'].mean()
        # loss = loss.detach()

        log_dict = dict(
            val_loss=loss,
            val_acc=acc,
            epoch=epoch,
        )

        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(log_dict, commit=True)