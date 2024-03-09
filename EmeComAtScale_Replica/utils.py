import json
import os

from transformers import GPT2TokenizerFast, ViTImageProcessor, ViTModel



def initialize_pretrained_models(img_checkpoint="google/vit-base-patch16-224-in21k"):
    """
    Initialize an image processor, and an image encoder from pretrained models.
    The parameters of the image encoder are frozen.

    Args:
        img_checkpoint (str): The checkpoint for the image model.

    Returns:
        ViTImageProcessor: The initialized image processor.
        ViTModel: The initialized image encoder with frozen parameters.
    """
    # moved to preprocessing, 2024-02-28 lg
    # image_processor = ViTImageProcessor.from_pretrained(img_checkpoint)
    img_encoder = ViTModel.from_pretrained(img_checkpoint)

    # freeze the encoder
    for param in img_encoder.parameters():
        param.requires_grad = False

    # moved to preprocessing, 2024-02-28 lg
    #return image_processor, img_encoder
    return img_encoder



def generate_vocab_file(vocab_size, filename="vocab.txt"):
    """
    Generate a vocabulary file with a given size.

    Args:
        vocab_size (int): The size of the vocabulary.
        filename (str): The name of the file to save the vocabulary.
    """

    vocab_dict={k:k for k in range(vocab_size)}
    with open(filename, "w") as f:
        json.dump(vocab_dict, f)

    # get file path

    filename = os.path.abspath(filename)

    return filename

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data")


    group.add_argument(
        "--data_subset", type=float, default=1.0, help="A percentage of the dataset to load (default: 1.0)"
    )

    group.add_argument(
        "--data_split",
        type=str,
        default="all",
        choices=["valid", "train"],
        help="Dataset split to load",
    )
    group.add_argument(
        "--distractors_num", type=int, default=3, help="Number of distractor images to use. -1 for none"
    )


    group.add_argument(
        "--num_workers", type=int, default=4, help="Workers used in the dataloader"
    )



def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module")

    group.add_argument(
        "--vision_chk",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        choices=["google/vit-base-patch16-224-in21k"],
        help="Checkpoint for pretrained Vit",
    )




def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=2048,
        help="Projection head's hidden dimension for image features",
    )
    group.add_argument(
        "--projection_output_dim",
        type=int,
        default=2048,
        help="Projection head's output dimension for image features",
    )

def get_loss_opts(parser):
    group = parser.add_argument_group("loss")

    group.add_argument(
        "--loss_temperature",
        type=float,
        default=0.1,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Similarity function used in loss",
    )


def get_common_opts(params)-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )



    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debugging mode: no wandb, no saving",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_loss_opts(parser)
    get_game_arch_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
