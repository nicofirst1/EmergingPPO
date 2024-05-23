"""
Transformer-based models for emergent communication
"""

from functools import partial
from typing import Iterator, Tuple

import torch
import torch.nn as nn
from egg.core import SenderReceiverContinuousCommunication
from torch.nn import Parameter
from transformers import (
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    PreTrainedTokenizerBase,
)
from transformers.generation import LogitsProcessorList


class GubleLogitsProcessor(nn.Module):
    """
    A module that applies the Gumbel-Softmax trick to the input scores.

    Args:
        temperature (float): The temperature for the Gumbel-Softmax trick. Default is 1.0.
        hard (bool): If True, the returned samples will be one-hot, otherwise they will be probabilities from the Gumbel-Softmax distribution. Default is True.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = True):
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, input_ids: torch.Tensor, next_token_logits) -> torch.Tensor:
        """
        Apply the Gumbel-Softmax trick to the input scores.

        Args:
            scores (torch.Tensor): The input scores/logits.

        Returns:
            torch.Tensor: The processed scores.
        """

        # check this for explenation https://github.com/pytorch/pytorch/issues/97851
        scores = torch.nn.functional.gumbel_softmax(
            next_token_logits, tau=self.temperature, hard=self.hard
        )

        # print("next_token_logits in GubleLogitsProcessor:\n", next_token_logits.deeper)
        # print("scores in GubleLogitsProcessor:\n", scores.deeper)

        return scores


class Sender(nn.Module):
    """
    Sender/receiver agent based on vision encoder decoder.

    Args:
        tokenizer (GPT2TokenizerFast): The tokenizer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int = 6,
        max_length: int = 6,
        gs_temperature: float = 1.0,
    ):
        super().__init__()

        self.tokenizer = tokenizer

        # Define logits processor

        logit_processor = GubleLogitsProcessor(temperature=gs_temperature)
        self.logit_processor = LogitsProcessorList([logit_processor])

        # init decoder
        config_decoder = GPT2Config(
            add_cross_attention=True,
            vocab_size=vocab_size,
        )
        self.decoder = GPT2LMHeadModel(config=config_decoder)

        # Force decoder to use cross attention
        self.decoder.prepare_inputs_for_generation = partial(
            self.prepare_inputs_for_generation,
            self.decoder.prepare_inputs_for_generation,
        )

        # Initialize weights for decoder
        self.decoder.init_weights()

        self.generation_config = GenerationConfig(
            # choose greedy decoding
            do_sample=False,
            num_beams=1,
            # stopping criteria
            max_length=max_length,
            # others
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @staticmethod
    def prepare_inputs_for_generation(
        orig_fn, input_ids: torch.Tensor, **model_kwargs
    ) -> dict:
        """
        Prepare the inputs for generation.

        Args:
            orig_fn (Callable): The original function for preparing inputs for generation.
            input_ids (torch.Tensor): The input IDs for the decoder.
            model_kwargs (dict): Additional keyword arguments for the model.

        Returns:
            dict: The prepared inputs for generation.
        """
        out = orig_fn(input_ids, **model_kwargs)

        if (
            "encoder_hidden_states" in model_kwargs
            and "encoder_hidden_states" not in out
        ):
            out["encoder_hidden_states"] = model_kwargs["encoder_hidden_states"]

        return out

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.decoder.parameters(recurse)

    @staticmethod
    def search_for_orig(decorated, orig_name):
        # hacky way to remove decorator
        for obj in (c.cell_contents for c in decorated.__closure__):
            if hasattr(obj, "__name__") and obj.__name__ == orig_name:
                return obj
            if hasattr(obj, "__closure__") and obj.__closure__:
                found = search_for_orig(obj, orig_name)
                if found:
                    return found
        return None

    def forward(self, sender_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Sender.

        Args:
            sender_input (torch.Tensor, optional): The input to the Sender. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The messages and scores.
        """

        # Forward pass through the encoder
        # enc_out = self.encoder(pixel_values=pixel_values)
        enc_out = sender_input
        # dim [batch_size, 197, 768]
        pooled_enc_out = enc_out.mean(dim=1)
        # dim [batch_size, 768]

        # remove the no_grad decorator to enable backpropagation
        generate_fn = self.search_for_orig(self.decoder.generate, "generate")

        # Forward generation through the decoder
        gen_out = generate_fn(
            # pass self since we lost it with the decorator
            self=self.decoder,
            # inputs=decoder_input_ids,
            generation_config=self.generation_config,
            # enables cross attention by passing the pooled encoder out
            encoder_hidden_states=enc_out,
            logits_processor=self.logit_processor,
        )

        messages = gen_out.sequences
        scores = torch.stack(gen_out.scores, dim=1)
        # dim [batch_size, max_length, vocab_size]

        # print ("messages :\n", messages.deeper)
        # print ("scores :\n", scores.deeper)
        # print ('\n\n')
        return messages, scores


class Receiver(nn.Module):
    """
    Sender/receiver agent based on vision encoder decoder
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        linear_dim: int = 123,
        vocab_size: int = 6,
    ):
        """
        Initialize the Receiver with an image encoder, tokenizer, and linear dimension.

        Args:
            tokenizer (GPT2TokenizerFast): The tokenizer.
            linear_dim (int): The linear dimension for the image and text encoders.
        """
        super().__init__()

        self.tokenizer = tokenizer

        # decoder
        config_decoder = GPT2Config(vocab_size=vocab_size)
        self.text_encoder = GPT2Model(config=config_decoder)

        # linear layers
        img_hidden_size = 768
        self.W_img = nn.Linear(img_hidden_size, linear_dim)
        self.W_txt = nn.Linear(self.text_encoder.embed_dim, linear_dim)
        self.text_embedding = nn.Linear(
            vocab_size, self.text_encoder.config.hidden_size
        )

    def pool_txt(self, h_read: torch.Tensor):
        """
        Pool the text read.

        Args:
            h_read (torch.Tensor): The text read.

        Returns:
            torch.Tensor: The pooled text read.
        """
        return h_read.mean(dim=1)

    def pool_img(self, h_image: torch.Tensor):
        """
        Pool the image.

        Args:
            h_image (torch.Tensor): The image.

        Returns:
            torch.Tensor: The pooled image.
        """
        return h_image.mean(dim=1)

    def forward(self, scores: torch.Tensor, receiver_input: torch.Tensor):
        """
        Forward pass through the Receiver.

        Args:
            scores (torch.Tensor): The sender' scores for message creation.
            receiver_input (torch.Tensor): The input to the Receiver.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The text and image encoder outputs.
        """

        # image encoding
        img_enc_out = receiver_input

        img_enc_out = self.pool_img(img_enc_out)
        img_enc_out = self.W_img(img_enc_out)

        # text encoding
        txt_enc_out = self.text_embedding(scores)
        txt_enc_out = self.pool_txt(txt_enc_out)
        txt_enc_out = self.W_txt(txt_enc_out)

        return txt_enc_out, img_enc_out


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)

    def forward(self, sender_input, labels, receiver_input, aux_input=None):

        message, scores = self.sender(sender_input)

        txt_enc_out, img_enc_out = self.receiver(scores, receiver_input)

        loss, aux_info = self.loss(img_enc_out, txt_enc_out)

        aux_info["img_id"] = aux_input

        aux_input = dict(
            scores=scores.detach().squeeze(1),
            txt_enc_out=txt_enc_out.detach(),
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=img_enc_out.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
