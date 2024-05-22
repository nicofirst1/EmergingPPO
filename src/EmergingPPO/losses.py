from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F


class NTXentLoss:
    """NTXentLoss as originally described in https://arxiv.org/abs/2002.05709.

    This loss is used in self-supervised learning setups and requires the two views of the input datapoint
    to be returned distinctly by Sender and Receiver.
    Note that this loss considers in-batch negatives and and negatives samples are taken within each agent
    datapoints i.e. each non-target element in sender_input and in receiver_input is considered a negative sample.

    >>> x_i = torch.eye(128)
    >>> x_j = torch.eye(128)
    >>> loss_fn = NTXentLoss()
    >>> loss, aux = loss_fn(None, x_i, None, x_j, None, None)
    >>> aux["acc"].mean().item()
    1.0
    >>> aux["acc"].shape
    torch.Size([256])
    >>> x_i = torch.eye(256)
    >>> x_j = torch.eye(128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    Traceback (most recent call last):
        ...
    RuntimeError: sender_output and receiver_output must be of the same shape, found ... instead
    >>> _ = torch.manual_seed(111)
    >>> x_i = torch.rand(128, 128)
    >>> x_j = torch.rand(128, 128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    >>> aux['acc'].mean().item() * 100  # chance level with a batch size of 128, 1/128 * 100 = 0.78125
    0.78125
    """

    def __init__(
        self,
        temperature: float = 1.0,
        similarity: str = "cosine",
        distractors: int = -1,
    ):
        self.temperature = temperature
        self.distractors = distractors

        similarities = {"cosine", "dot"}
        assert (
            similarity.lower() in similarities
        ), f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    @staticmethod
    def ntxent_loss(
        sender_output: torch.Tensor,
        receiver_output: torch.Tensor,
        temperature: float = 1.0,
        similarity: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if sender_output.shape != receiver_output.shape:
            raise RuntimeError(
                f"sender_output and receiver_output must be of the same shape, "
                f"found {sender_output.shape} and {receiver_output.shape} instead"
            )
        batch_size = sender_output.shape[0]

        input = torch.cat((sender_output, receiver_output), dim=0)

        if similarity == "cosine":
            similarity_f = torch.nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / temperature
            )
        elif similarity == "dot":
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )

        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(
            0
        )

        negative_samples = similarity_matrix[mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="none") / 2

        acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        return loss, {"acc": acc}

    def modified_ntxent_loss(
        self, text_embeddings, image_embeddings, correct_image, temperature=1.0
    ):
        """
        Custom contrastive loss function to align text embeddings with corresponding
        correct image embeddings and differentiate them from other images.

        Args:
        text_embeddings: Tensor of shape [batch_size, features].
        image_embeddings: Tensor of shape [batch_size, num_images, features].
        correct_image: Long tensor of shape [batch_size] with indices of the correct images.
        temperature: A float for temperature scaling.

        Returns:
        loss: A scalar tensor representing the computed loss.
        """

        batch_size, num_images, _ = image_embeddings.shape

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=2)

        # Reshape and tile text embeddings to compare with each image
        text_embeddings_tiled = text_embeddings.unsqueeze(1).repeat(1, num_images, 1)

        # Compute similarity scores
        similarities = (
            torch.sum(text_embeddings_tiled * image_embeddings, dim=2) / temperature
        )

        # Create labels - correct image will have label '1', others '0'
        labels = torch.zeros(batch_size, num_images, device=text_embeddings.device)
        labels[torch.arange(batch_size), correct_image] = 1

        # Convert similarities to logits
        logits = F.log_softmax(similarities, dim=1)

        # get positive and negative samples
        positive_samples = logits[range(batch_size), correct_image]
        negative_samples = logits[labels == 0]

        # Compute the loss
        loss = positive_samples.mean() - negative_samples.mean()

        predicted_image = torch.argmax(similarities, dim=1)
        accuracy = (predicted_image == correct_image).float().mean()

        aux_info = {
            "acc": accuracy,
            "acc-random": accuracy - 1 / num_images,
            #   "predicted_image": predicted_image,
        }

        aux_info = {k: v.unsqueeze(0) for k, v in aux_info.items()}

        print(accuracy)

        return loss, aux_info

    def __call__(
        self,
        img_encoding,
        text_encoding,
        sender_message,
        sender_socres,
        labels,
    ):

        if self.distractors < 1:
            return self.ntxent_loss(
                img_encoding,
                text_encoding,
                temperature=self.temperature,
                similarity=self.similarity,
            )
        else:
            return self.modified_ntxent_loss(
                img_encoding,
                text_encoding,
                labels,
                temperature=self.temperature,
            )
