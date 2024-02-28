import torch
from datasets import load_dataset
from egg.core import Trainer, ProgressBarLogger
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, MaxLengthCriteria
import wandb

try:
    from EmeComAtScale_Replica.data import emecom_map, custom_collate_fn
    from EmeComAtScale_Replica.losses import NTXentLoss
    from EmeComAtScale_Replica.utils import initialize_pretrained_models, generate_vocab_file, get_common_opts
    from EmeComAtScale_Replica.utils_logs import CustomWandbLogger
except ModuleNotFoundError:
    from data import emecom_map, custom_collate_fn
    from losses import NTXentLoss
    from utils import initialize_pretrained_models, generate_vocab_file, get_common_opts
    from utils_logs import CustomWandbLogger

from models import Sender, Receiver, EmComSSLSymbolGame


def main(args):
    opts = get_common_opts(params=args)

    # add mac m1
    # opts.device = torch.device("mps")
    print(f"{opts}\n")

    image_processor, img_encoder = initialize_pretrained_models(opts.vision_chk)

    # tokenizer
    vocab_file = generate_vocab_file(opts.vocab_size)
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    tokenizer.bos_token_id = tokenizer.cls_token_id

    # stopping criteria as maxlength for decoder
    stopping_criteria = MaxLengthCriteria(max_length=opts.max_len)

    sender = Sender(img_encoder=img_encoder, tokenizer=tokenizer, vocab_size=opts.vocab_size,
                    stopping_criteria=stopping_criteria, gs_temperature=opts.gs_temperature)
    receiver = Receiver(img_encoder=img_encoder, tokenizer=tokenizer, vocab_size=opts.vocab_size,
                        linear_dim=opts.projection_output_dim)

    sender.to(opts.device)
    receiver.to(opts.device)

    # todo : use different losses
    loss = NTXentLoss(
        temperature=opts.loss_temperature,
        similarity=opts.similarity,
        distractors=opts.distractors_num,
    )

    game = EmComSSLSymbolGame(
        sender=sender,
        receiver=receiver,
        loss=loss,
        distractors=opts.distractors_num
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

    # todo: load all splits
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')

    # filter all images where the mode is not RBG
    dataset = dataset.filter(lambda e: e["image"].mode == "RGB")

    # #todo: comment this out
    # dataset = dataset.filter(lambda e, i: i < 105, with_indices=True)

    # preprocess the images
    dataset = dataset.map(emecom_map, batched=True, remove_columns=["image"],
                          fn_kwargs={"num_distractors": opts.distractors_num, "image_processor": image_processor},
                          num_proc=opts.num_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,

    )

    ## CALLBACKS
    progress_bar = ProgressBarLogger(n_epochs=opts.n_epochs,
                                     train_data_len=len(dataloader))


    wandb_logger = CustomWandbLogger(entity='emergingtransformer',
                                     project='EmergingPPO',
                                     opts=opts,
                                     # todo: comment out
                                    # mode="offline"
                                     )

    wandb.watch(game)

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        # optimizer_scheduler=optimizer_scheduler,
        train_data=dataloader,
        callbacks=[progress_bar, wandb_logger],
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
