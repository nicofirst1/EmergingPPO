from typing import Callable, Union

import numpy as np
import torch
from egg.core import Interaction
from egg.core.callbacks import WandbLogger
from egg.core.language_analysis import TopographicSimilarity
from measures import Message, Messages, normalized_editdistance, pairwise_dedup
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr

import wandb


class CustomTopographicSimilarity(TopographicSimilarity):
    @staticmethod
    def compute_topsim(
        meanings: torch.Tensor,
        messages: Messages,
        meaning_distance_fn: Union[str, Callable] = "hamming",
        message_distance_fn: Union[str, Callable] = "edit",
    ) -> float:

        distances = {
            # original
            # "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
            "edit": normalized_editdistance,  # my version
            "cosine": distance.cosine,
            "hamming": distance.hamming,
            "jaccard": distance.jaccard,
            "euclidean": distance.euclidean,
        }

        meaning_distance_fn = (
            distances.get(meaning_distance_fn, None)
            if isinstance(meaning_distance_fn, str)
            else meaning_distance_fn
        )
        message_distance_fn = (
            distances.get(message_distance_fn, None)
            if isinstance(message_distance_fn, str)
            else message_distance_fn
        )

        assert (
            meaning_distance_fn and message_distance_fn
        ), f"Cannot recognize {meaning_distance_fn} \
            or {message_distance_fn} distances"

        # Orig code
        # meaning_dist = distance.pdist(meanings, meaning_distance_fn)
        # message_dist = distance.pdist(messages, message_distance_fn)

        assert len(messages) == len(meanings)
        meaning_dist = np.array(list(pairwise_dedup(meaning_distance_fn, meanings)))
        message_dist = np.array(list(pairwise_dedup(message_distance_fn, messages)))

        # Raviv 2019 uses pearson
        # topsim, __pval = pearsonr(meaning_dist, message_dist)

        # Egg uses Spearman, let's go with it
        topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

        return topsim

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:

        if logs == Interaction.empty():
            return

        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = [msg.tolist() for msg in messages]
        sender_input = torch.flatten(logs.sender_input, start_dim=1)

        print("sender_input.size()", sender_input.size())
        print("len(messages)", len(messages))
        print("messages[0]", messages[0])
        topsim = self.compute_topsim(
            sender_input,
            messages,
            self.sender_input_distance_fn,
            self.message_distance_fn,
        )

        logs.aux["topsim"] = torch.as_tensor(topsim)

        wandb_dict = {f"{mode}/topsim": topsim, "epoch": epoch}
        wandb.log(wandb_dict)

        print(wandb_dict, flush=True)


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

        batch_acc_scores = logs.aux["acc"]
        print("batch_acc_scores", batch_acc_scores.size())
        print("batch_acc_scores.size()", batch_acc_scores.size())
        acc = batch_acc_scores.mean()
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
            f"{split}/loss": loss,
            f"{split}/acc": acc,
            "epoch": epoch,
        }

        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(log_dict, commit=True)
