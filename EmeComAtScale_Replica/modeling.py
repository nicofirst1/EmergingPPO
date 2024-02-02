"""
Transformer-based models for emergent communication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core import NTXentLoss

from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor


class ViTGPT2Agent(nn.Module):

    """Sender/receiver agent based on vision encoder decoder"""

    def __init__(self):
        """TODO: to be defined."""
        nn.Module.__init__(self)

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        # We can use our own here.
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        self.image_processor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        # TODO fix dimensions according to last hidden of ViT, GPT-2, respectively
        self.W_img = nn.Linear(123,123)  # dummy dimensions
        self.W_txt = nn.Linear(234,123)


    def pool_txt(self, h_read):
        # TODO pool on sequence dimension, or special token?
        return h_read.mean(dim=1)

    def pool_img(self, h_image):
        # TODO pool on patches dimension, or special token?
        return h_image.mean(dim=1)

    def forward(
        self,
        sender_input=None,
        aux_input=None,  # input to both sender/receiver, unused atm
        message=None,
        receiver_input=None
    ):
        if message is None:
            # Act as Sender (~Image captioning)
            pixel_values = self.image_processor(
                sender_input, return_tensor="pt"
            ).pixel_values

            message_logits = self.model(pixel_values=pixel_values)

            # Inference/logging actual messages:
            # generated_message = self.model.generate(pixel_values)  # Greedy Decoding
            # generated_message_decoded = tokenizer.batch_decode(generated_message, skip_special_tokens=True)
            outputs = message_logits

            # TODO do gumbel softmax'ing either here or client-side
        else:
            # Act as Receiver (~CLIP/Contrastive learning)
            assert receiver_input is not None
            h_txt_sequence = self.model.decoder(input_ids=message)
            h_img_patches = self.image_processor(pixel_values=receiver_inputs)

            h_txt = self.W_txt(self.pool_txt(h_txt_sequence))
            h_img = self.W_img(self.pool_img(h_img_patches))

            # TODO make sure this works well with multiple distractors
            con_loss, con_acc = NTXentLoss.ntxent_loss(
                h_img, h_txt, temperature=1.0, similarity="cosine"
            )
            outputs = con_loss

        return outputs


