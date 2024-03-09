"""
Transformer-based models for emergent communication
"""
from functools import partial
from typing import Iterator, Tuple, Optional

import torch
import torch.nn as nn
from egg.core import SenderReceiverContinuousCommunication
from torch.nn import Parameter
from transformers import GPT2Config, MaxLengthCriteria, GPT2Model, GPT2LMHeadModel, ViTModel, \
    GPT2TokenizerFast, StoppingCriteriaList, PreTrainedTokenizerBase


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

    def forward(self, scores: torch.Tensor, next_token_logits) -> torch.Tensor:
        """
        Apply the Gumbel-Softmax trick to the input scores.

        Args:
            scores (torch.Tensor): The input scores/logits.

        Returns:
            torch.Tensor: The processed scores.
        """
        return torch.nn.functional.gumbel_softmax(next_token_logits, tau=self.temperature, hard=self.hard)


class Sender(nn.Module):
    """
        Sender/receiver agent based on vision encoder decoder.

        Args:
            img_encoder (ViTModel): The image encoder.
            tokenizer (GPT2TokenizerFast): The tokenizer.
        """

    def __init__(self, img_encoder: ViTModel, tokenizer: PreTrainedTokenizerBase,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 vocab_size: int = 6, gs_temperature: float = 1.0):
        super().__init__()

        self.tokenizer = tokenizer
        self.encoder = img_encoder

        # Define stopping criteria and logits processor
        self.stopping_criteria = MaxLengthCriteria(max_length=6) if stopping_criteria is None else stopping_criteria
        self.logit_processor = GubleLogitsProcessor(temperature=gs_temperature)

        # init decoder
        config_decoder = GPT2Config(add_cross_attention=True,
                                    vocab_size=vocab_size, )
        self.decoder = GPT2LMHeadModel(config=config_decoder)

        # Force decoder to use cross attention
        self.decoder.prepare_inputs_for_generation = partial(self.prepare_inputs_for_generation,
                                                             self.decoder.prepare_inputs_for_generation)

        # Initialize weights for decoder
        self.decoder.init_weights()

    @staticmethod
    def prepare_inputs_for_generation(orig_fn, input_ids: torch.Tensor, **model_kwargs) -> dict:
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

        if "encoder_hidden_states" in model_kwargs and "encoder_hidden_states" not in out:
            out["encoder_hidden_states"] = model_kwargs["encoder_hidden_states"]

        return out

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.decoder.parameters(recurse)

    def forward(self, sender_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Sender.

        Args:
            sender_input (torch.Tensor, optional): The input to the Sender. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The messages and scores.
        """
        pixel_values = sender_input

        # Forward pass through the encoder
        enc_out = self.encoder(pixel_values=pixel_values)

        # Create the decoder input
        bos_token_id = self.tokenizer.bos_token_id
        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
        decoder_input_ids = decoder_input_ids.to(pixel_values.device)

        # Forward generation through the decoder
        gen_out = self.decoder.greedy_search(
            decoder_input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            encoder_outputs=enc_out,
            encoder_hidden_states=enc_out.last_hidden_state,  # enables cross attention
            stopping_criteria=self.stopping_criteria,
            return_dict_in_generate=True,
            logits_processor=self.logit_processor,
        )

        messages = gen_out.sequences
        scores = torch.stack(gen_out.scores, dim=1)

        return messages, scores


class Receiver(nn.Module):
    """
    Sender/receiver agent based on vision encoder decoder
    """

    def __init__(self, img_encoder: ViTModel, tokenizer: PreTrainedTokenizerBase, linear_dim: int = 123,
                 vocab_size: int = 6):
        """
        Initialize the Receiver with an image encoder, tokenizer, and linear dimension.

        Args:
            img_encoder (ViTModel): The image encoder.
            tokenizer (GPT2TokenizerFast): The tokenizer.
            linear_dim (int): The linear dimension for the image and text encoders.
        """
        super().__init__()

        self.img_encoder = img_encoder
        self.tokenizer = tokenizer

        # decoder
        config_decoder = GPT2Config(vocab_size=vocab_size)
        self.text_encoder = GPT2Model(config=config_decoder)

        # linear layers
        self.W_img = nn.Linear(self.img_encoder.config.hidden_size, linear_dim)
        self.W_txt = nn.Linear(self.text_encoder.embed_dim, linear_dim)
        self.text_embedding = nn.Linear(vocab_size, self.text_encoder.config.hidden_size)

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
        pixel_values = receiver_input

        if pixel_values.ndim > 4:
            # using distractors [batch size, distractors, img_dim x3]
            img_enc_out = [self.img_encoder(pixel_values=pv) for pv in pixel_values]
            img_enc_out = [self.pool_img(e.last_hidden_state) for e in img_enc_out]
            img_enc_out = [self.W_img(e) for e in img_enc_out]
            img_enc_out = torch.stack(img_enc_out)
        else:
            # no distractors [batch size, img_dim x3]

            img_enc_out = self.img_encoder(pixel_values).last_hidden_state
            img_enc_out = self.pool_img(img_enc_out)
            img_enc_out = self.W_img(img_enc_out)

            # text encoding
        txt_enc_out = self.text_embedding(scores)
        txt_enc_out = self.pool_txt(txt_enc_out)
        txt_enc_out = self.W_txt(txt_enc_out)

        return txt_enc_out, img_enc_out


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, distractors, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)
        self.distractors = distractors

    def forward(self, sender_input, labels, receiver_input, aux_input=None):
        if self.distractors < 1:
            # if no distractors present input is the same for both
            sender_input = receiver_input

        message, scores = self.sender(sender_input)
        txt_enc_out, img_enc_out = self.receiver(scores, receiver_input)

        loss, aux_info = self.loss(
            img_enc_out, txt_enc_out, message, scores, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=None,
            receiver_output=scores.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
