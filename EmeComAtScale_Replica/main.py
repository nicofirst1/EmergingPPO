import torch
import wandb
from egg.core import Trainer, ConsoleLogger
from egg.core.interaction import IntervalLoggingStrategy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, MaxLengthCriteria

try:
    from EmeComAtScale_Replica.data import (
        custom_collate_fn,
        load_and_preprocess_dataset,
    )
    from EmeComAtScale_Replica.losses import NTXentLoss
    from EmeComAtScale_Replica.utils import (
        initialize_pretrained_models,
        generate_vocab_file,
        get_common_opts,
    )
    from EmeComAtScale_Replica.utils_logs import (
        CustomWandbLogger,
        CustomTopographicSimilarity,
    )
except ModuleNotFoundError:
    from data import custom_collate_fn, load_and_preprocess_dataset
    from losses import NTXentLoss
    from utils import initialize_pretrained_models, generate_vocab_file, get_common_opts
    from utils_logs import CustomWandbLogger, CustomTopographicSimilarity

from models import Sender, Receiver, EmComSSLSymbolGame


def main(args):
    opts = get_common_opts(params=args)

    # add mac m1
    # opts.device = torch.device("mps")
    print(f"{opts}\n")

    img_encoder = initialize_pretrained_models(opts.vision_chk)

    # tokenizer
    vocab_file = generate_vocab_file(opts.vocab_size)
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    tokenizer.bos_token_id = tokenizer.cls_token_id

    # stopping criteria as maxlength for decoder
    stopping_criteria = MaxLengthCriteria(max_length=opts.max_len)

    sender = Sender(
        img_encoder=img_encoder,
        tokenizer=tokenizer,
        vocab_size=opts.vocab_size,
        stopping_criteria=stopping_criteria,
        gs_temperature=opts.gs_temperature,
    )
    receiver = Receiver(
        img_encoder=img_encoder,
        tokenizer=tokenizer,
        vocab_size=opts.vocab_size,
        linear_dim=opts.projection_output_dim,
    )

    sender.to(opts.device)
    receiver.to(opts.device)

    # todo : use different losses
    loss = NTXentLoss(
        temperature=opts.loss_temperature,
        similarity=opts.similarity,
        distractors=opts.distractors_num,
    )

    logging_strategy = IntervalLoggingStrategy(
        store_sender_input=True,
        store_receiver_input=False,
        store_labels=True,
        store_aux_input=True,
        store_message=True,
        store_receiver_output=False,
        store_message_length=False,
        log_interval=opts.log_interval,
    )

    game = EmComSSLSymbolGame(
        sender=sender,
        receiver=receiver,
        loss=loss,
        distractors=opts.distractors_num,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
    )

    model_parameters = list(sender.parameters()) + list(receiver.parameters())

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    # optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=5
    # )

    dataset = load_and_preprocess_dataset("Maysee/tiny-imagenet",
                                          opts.data_split,
                                          opts.vision_chk,
                                          distractors_num=opts.distractors_num,
                                          data_subset=opts.data_subset
                                          )

    train_dataloader = DataLoader(
        dataset[0],
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    if len(dataset) > 1:
        valid_dataloader = DataLoader(
            dataset[1],
            batch_size=opts.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
    else:
        valid_dataloader = None

    ## CALLBACKS
    console_logger = ConsoleLogger(print_train_loss=True, as_json=True)

    topsim = CustomTopographicSimilarity(
        sender_input_distance_fn="euclidean",
        message_distance_fn="edit",
        compute_topsim_train_set=True,
        compute_topsim_test_set=True,
        is_gumbel=True,
    )

    wandb_logger = CustomWandbLogger(
        entity="emergingtransformer",
        project="EmergingPPO",
        opts=opts,
        mode="offline" if opts.debug else "online",
    )

    # wandb.watch(game)
    #wandb.watch((sender, receiver), log_freq=1000, log_graph=False)
    # 2024-04-10, lg: Run finished with no logs uploaded to wb -> we have our custom log freq now, do we need wb's?
    wandb.watch((sender, receiver), log_graph=False)

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        # optimizer_scheduler=optimizer_scheduler,
        train_data=train_dataloader,
        validation_data=valid_dataloader,
        callbacks=[topsim, wandb_logger, console_logger],
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
