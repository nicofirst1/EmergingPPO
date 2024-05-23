# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data")

    group.add_argument(
        "--data_subset",
        type=float,
        default=None,
        help="A percentage of the dataset to load (default: None -> load all data)",
    )

    group.add_argument(
        "--data_split",
        type=str,
        default="all",
        choices=["valid", "train"],
        help="Dataset split to load",
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


def args_fixer(opts):
    """
    Check and potentially fix the arguments
    """

    if opts.data_subset is not None:
        if opts.data_subset > 1:
            # cast to int if data_subset is a value
            opts.data_subset = int(opts.data_subset)
        else:
            # if perc then assert that it is in [0.0,1.0]
            opts.data_subset = float(opts.data_subset)
            assert opts.data_subset <= 1, "data_subset should be a value in [0.0,1.0]"

        assert opts.data_subset > 0, "data_subset should be > 0!"


def get_log_opts(parser):
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debugging mode: no wandb, no saving",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval for logging",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="logs/",
        help="Path to save the model",
    )

    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=50,
        help="Path to save the model",
    )

    parser.add_argument(
        "--sender_input_distance_fn",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "edit", "hamming", "jaccard"],
        help="Distance function used to compute the similarity between sender input",
    )
    parser.add_argument(
        "--message_distance_fn",
        type=str,
        default="edit",
        choices=["cosine", "euclidean", "edit", "hamming", "jaccard"],
        help="Distance function used to compute the similarity between messages",
    )


def get_common_opts(params) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_loss_opts(parser)
    get_game_arch_opts(parser)
    get_log_opts(parser)

    opts = core.init(arg_parser=parser, params=params)

    args_fixer(opts)

    return opts
