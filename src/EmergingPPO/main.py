import torch
import wandb
from egg.core import ConsoleLogger, Trainer
from egg.core.batch import Batch
from egg.core.interaction import IntervalLoggingStrategy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, MaxLengthCriteria


from EmergingPPO.data import custom_collate_fn, load_and_preprocess_dataset
from EmergingPPO.losses import NTXentLoss
from EmergingPPO.utils import initialize_pretrained_models, generate_vocab_file, get_common_opts
from EmergingPPO.utils_logs import CustomWandbLogger, CustomTopographicSimilarity

from EmergingPPO.models import EmComSSLSymbolGame, Receiver, Sender
from EmergingPPO.saver import ModelSaverCallback


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

    train_logging_strategy = IntervalLoggingStrategy(
        store_sender_input=True,
        store_receiver_input=False,
        store_labels=True,
        store_aux_input=True,
        store_message=True,
        store_receiver_output=False,
        store_message_length=False,
        log_interval=opts.log_interval,
    )

    test_logging_strategy = IntervalLoggingStrategy(
        store_sender_input=True,
        store_receiver_input=False,
        store_labels=True,
        store_aux_input=True,
        store_message=True,
        store_receiver_output=False,
        store_message_length=False,
        log_interval=1,
    )

    game = EmComSSLSymbolGame(
        sender=sender,
        receiver=receiver,
        loss=loss,
        distractors=opts.distractors_num,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
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

    dataset = load_and_preprocess_dataset(
        "Maysee/tiny-imagenet",
        opts.data_split,
        opts.vision_chk,
        distractors_num=opts.distractors_num,
        data_subset=opts.data_subset,
    )

    print(f"Number of datasets: {len(dataset)} (should be 2 if data_split='all'")
    print(f"Batch size: {opts.batch_size}")

    assert opts.batch_size > 1

    # Actually treated as list-of-datasets, either 1 or 2
    assert len(dataset) == 1 or len(dataset) == 2

    print("Train dataset", dataset[0])
    print("Valid dataset", dataset[1])

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

    ## DUMMY SWEEPS
    print("Dummy sweep train loader")

    for i, batch in enumerate(train_dataloader):
        # Same as in egg's trainer
        if not isinstance(batch, Batch):
            batch = Batch(*batch)
        print("Sender input size", batch.sender_input.size())
        print("Labels size", batch.labels.size())
        print("Receiver input size", batch.receiver_input.size())
        print("Aux input size", batch.aux_input.size())
        if i > 5:
            break

    print("Dummy sweep valid loader")
    for i, batch in enumerate(valid_dataloader):
        # Same as in egg's trainer
        if not isinstance(batch, Batch):
            batch = Batch(*batch)
        print("Sender input size", batch.sender_input.size())
        print("Labels size", batch.labels.size())
        print("Receiver input size", batch.receiver_input.size())
        print("Aux input size", batch.aux_input.size())
        if i > 5:
            break

    ## CALLBACKS
    console_logger = ConsoleLogger(print_train_loss=True, as_json=True)

    topsim = CustomTopographicSimilarity(
        sender_input_distance_fn="euclidean",
        message_distance_fn="edit",
        compute_topsim_train_set=False,
        compute_topsim_test_set=True,
        is_gumbel=False,  # message should be already argmax'ed, 2024-04-16 lg
    )

    wandb_logger = CustomWandbLogger(
        entity="emergingtransformer",
        project="EmergingPPO",
        opts=opts,
        mode="offline" if opts.debug else "online",
    )

    speaker_saver = ModelSaverCallback(
        sender, opts.save_path, "sender", save_every_n_epochs=opts.save_every_n_epochs
    )

    # wandb.watch(game)
    # wandb.watch((sender, receiver), log_freq=1000, log_graph=False)
    # 2024-04-10, lg: Run finished with no logs uploaded to wb -> we have our custom log freq now, do we need wb's?
    wandb.watch((sender, receiver), log_graph=False)

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        # optimizer_scheduler=optimizer_scheduler,
        train_data=train_dataloader,
        validation_data=valid_dataloader,
        callbacks=[topsim, wandb_logger, console_logger, speaker_saver],
    )

    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
