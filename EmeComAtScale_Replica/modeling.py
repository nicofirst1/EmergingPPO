"""
Transformer-based models for emergent communication
"""
from functools import partial

import torch
import torch.nn as nn

from egg.core import NTXentLoss

from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor, ViTConfig, \
    VisionEncoderDecoderConfig, GPT2Config, MaxLengthCriteria, LogitsProcessor, ViTModel, GPT2Model


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
            message=None,
            receiver_input=None
    ):
        if message is None:
            # Act as Sender (~Image captioning)
            pixel_values = self.image_processor(
                sender_input, return_tensor="pt"
            ).pixel_values

            pixel_values = torch.tensor(pixel_values)

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


class GubleLogitsProcessor(LogitsProcessor):

    def __init__(self, temperature=1.0, hard=True):
        self.temperature = temperature
        self.hard = hard

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nn.functional.gumbel_softmax(scores, tau=self.temperature, hard=self.hard)


def prepare_inputs_for_generation(orig_fn, input_ids, **model_kwargs):

    out=orig_fn(input_ids, **model_kwargs)

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
        self.model.decoder.prepare_inputs_for_generation=partial(prepare_inputs_for_generation,     self.model.decoder.prepare_inputs_for_generation)

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
            message=None,
            receiver_input=None
    ):
        pixel_values = self.image_processor(
            sender_input, return_tensor="pt"
        ).pixel_values

        # cast to tensor
        pixel_values = torch.tensor(pixel_values)
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
        #todo: check if xattention by checking if dec has enc out
        gen_out = self.model.decoder.greedy_search(
            decoder_input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            encoder_outputs=enc_out,
            encoder_hidden_states=enc_out.last_hidden_state, # enables Xattention
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            logits_processor=logit_processor,
        )

        messages = gen_out.sequences
        scores=torch.stack(gen_out.scores, dim =1)
        # TODO: something with the scores

        return messages, scores


class Receiver(nn.Module):
    """Sender/receiver agent based on vision encoder decoder"""

    def __init__(self, tokenizer, image_processor):
        """TODO: to be defined."""
        nn.Module.__init__(self)

        config_decoder = GPT2Config()

        #todo: freeze vit
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
            sender_input=None,
            scores=None,  # input to both sender/receiver, unused atm
            message=None,
            receiver_input=None
    ):
        pixel_values = [self.image_processor(ri, return_tensor="pt").pixel_values for ri in receiver_input]
        pixel_values = [torch.tensor(pv) for pv in pixel_values]
        # [batch_size,distractors, channels, height, width]

        # forward pass through the encoder
        img_enc_out = [self.img_encoder(pixel_values=pv) for pv in pixel_values]
        img_enc_out = [self.pool_img(e.last_hidden_state) for e in img_enc_out]
        img_enc_out = [self.W_img(e) for e in img_enc_out]
        img_enc_out = torch.stack(img_enc_out)

        # encode the message (one hot encoding)
        #txt_enc_out = self.text_encoder(input_ids=message)
        txt_enc_out=self.text_embedding(scores)
        txt_enc_out=self.pool_txt(txt_enc_out)
        txt_enc_out=self.W_txt(txt_enc_out)


        return_dict = {
            "img_enc_out": img_enc_out,
            "txt_enc_out": txt_enc_out
        }

        return return_dict
