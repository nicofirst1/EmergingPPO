"""
Transformer-based models for emergent communication
"""
from functools import partial

import torch
import torch.nn as nn
from egg.core import SenderReceiverContinuousCommunication
from transformers import VisionEncoderDecoderModel, ViTConfig, \
    VisionEncoderDecoderConfig, GPT2Config, MaxLengthCriteria, LogitsProcessor, ViTModel, GPT2Model


class GubleLogitsProcessor(LogitsProcessor):

    def __init__(self, temperature=1.0, hard=True):
        self.temperature = temperature
        self.hard = hard

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nn.functional.gumbel_softmax(scores, tau=self.temperature, hard=self.hard)


def prepare_inputs_for_generation(orig_fn, input_ids, **model_kwargs):
    out = orig_fn(input_ids, **model_kwargs)

    if "encoder_hidden_states" in model_kwargs and "encoder_hidden_states" not in out:
        out["encoder_hidden_states"] = model_kwargs["encoder_hidden_states"]

    return out


class Sender(nn.Module):
    """Sender/receiver agent based on vision encoder decoder"""

    def __init__(self, tokenizer, image_processor):
        """TODO: to be defined."""
        nn.Module.__init__(self)

        config_encoder = ViTConfig.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        config_decoder = GPT2Config(
            add_cross_attention=True,

        )
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        self.model = VisionEncoderDecoderModel(config=config)

        # force decoder to use xattention
        self.model.decoder.prepare_inputs_for_generation = partial(prepare_inputs_for_generation,
                                                                   self.model.decoder.prepare_inputs_for_generation)

        # # Initialize weights for decoder
        # self.model.decoder.init_weights()

        # We can use our own here.
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # TODO fix dimensions according to last hidden of ViT, GPT-2, respectively
        self.W_img = nn.Linear(123, 123)  # dummy dimensions
        self.W_txt = nn.Linear(234, 123)

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

    ):

        if len(sender_input.shape) < 4:
            pixel_values = self.image_processor(
                sender_input, return_tensor="pt"
            ).pixel_values

            # cast to tensor
            pixel_values = torch.tensor(pixel_values)
        else:
            pixel_values = sender_input

        # forward pass through the encoder
        enc_out = self.model.encoder(pixel_values=pixel_values)

        # create the decoder input
        bos_token_id = self.tokenizer.bos_token_id
        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long
        )

        # define stopping criteria and logits processor
        stopping_criteria = MaxLengthCriteria(
            max_length=6
        )
        logit_processor = GubleLogitsProcessor(temperature=1.0)

        # forward generation through the decoder
        # todo: check if xattention by checking if dec has enc out
        gen_out = self.model.decoder.greedy_search(
            decoder_input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            encoder_outputs=enc_out,
            encoder_hidden_states=enc_out.last_hidden_state,  # enables Xattention
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            logits_processor=logit_processor,
        )

        messages = gen_out.sequences
        scores = torch.stack(gen_out.scores, dim=1)

        return messages, scores


class Receiver(nn.Module):
    """Sender/receiver agent based on vision encoder decoder"""

    def __init__(self, tokenizer, image_processor):
        """TODO: to be defined."""
        nn.Module.__init__(self)

        config_decoder = GPT2Config()

        # todo: freeze vit
        self.img_encoder = ViTModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.text_encoder = GPT2Model(config=config_decoder)

        # We can use our own here.
        self.tokenizer = tokenizer

        self.image_processor = image_processor

        # TODO fix dimensions according to last hidden of ViT, GPT-2, respectively
        self.W_img = nn.Linear(self.img_encoder.config.hidden_size, 123)  # dummy dimensions
        self.W_txt = nn.Linear(self.text_encoder.embed_dim, 123)
        self.text_embedding = nn.Linear(self.tokenizer.vocab_size, self.text_encoder.config.hidden_size)

    def pool_txt(self, h_read):
        # TODO pool on sequence dimension, or special token?
        return h_read.mean(dim=1)

    def pool_img(self, h_image):
        # TODO pool on patches dimension, or special token?
        return h_image.mean(dim=1)

    def forward(
            self,
            scores,
            receiver_input
    ):
        if len(receiver_input.shape) < 5:

            pixel_values = [self.image_processor(ri, return_tensor="pt").pixel_values for ri in receiver_input]
            pixel_values = [torch.tensor(pv) for pv in pixel_values]
        else:
            pixel_values = receiver_input

        # [batch_size,distractors, channels, height, width]

        # forward pass through the encoder
        img_enc_out = [self.img_encoder(pixel_values=pv) for pv in pixel_values]
        img_enc_out = [self.pool_img(e.last_hidden_state) for e in img_enc_out]
        img_enc_out = [self.W_img(e) for e in img_enc_out]
        img_enc_out = torch.stack(img_enc_out)

        # encode the message (one hot encoding)
        # txt_enc_out = self.text_encoder(input_ids=message)
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

        loss, aux_info = self.loss.modified_ntxent_loss(
            txt_enc_out,img_enc_out,labels
        )

        if hasattr(self.sender, "temperature"):
            if isinstance(self.sender.temperature, torch.nn.Parameter):
                temperature = self.sender.temperature.detach()
            else:
                temperature = torch.Tensor([self.sender.temperature])
            aux_info["temperature"] = temperature

        if not self.training:
            aux_info["message"] = message


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
