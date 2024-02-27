import torch
from datasets import load_dataset
from egg.core import Trainer, ProgressBarLogger
from egg.zoo.emcom_as_ssl.utils import get_common_opts
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, BertTokenizerFast

try:
    from EmeComAtScale_Replica.data import emecom_map, custom_collate_fn
    from EmeComAtScale_Replica.losses import NTXentLoss
    from EmeComAtScale_Replica.utils import initialize_pretrained_models, generate_vocab_file
except ModuleNotFoundError:
    from data import emecom_map, custom_collate_fn
    from losses import NTXentLoss
    from utils import initialize_pretrained_models, generate_vocab_file

from models import Sender, Receiver, EmComSSLSymbolGame


def main(args):
    opts = get_common_opts(params=args)

    # add mac m1
    opts.device=  torch.device("mps")

    print(f"{opts}\n")

    image_processor, img_encoder = initialize_pretrained_models()

    # tokenizer
    vocab_file=generate_vocab_file(opts.vocab_size)
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    tokenizer.bos_token_id = tokenizer.cls_token_id


    sender = Sender(img_encoder=img_encoder, tokenizer=tokenizer, vocab_size=opts.vocab_size)
    receiver = Receiver(img_encoder=img_encoder, tokenizer=tokenizer,vocab_size=opts.vocab_size)

    sender.to(opts.device)
    receiver.to(opts.device)

    loss = NTXentLoss(
        temperature=1,
        similarity="cosine",
    )

    game = EmComSSLSymbolGame(
        sender,
        receiver,
        loss,
    )

    model_parameters = list(sender.parameters()) + list(receiver.parameters())

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=0.0001,
        momentum=0.9,
    )
    # optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=5
    # )

    dataset = load_dataset('Maysee/tiny-imagenet', split='train')

    # filter all images where the mode is not RBG
    dataset = dataset.filter(lambda e: e["image"].mode == "RGB")

    # preprocess the images
    dataset = dataset.map(emecom_map, batched=True, remove_columns=["image"],
                          fn_kwargs={"num_distractors": 3, "image_processor": image_processor},num_proc=2
                          )


    dataloader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,

    )

    progress_bar = ProgressBarLogger(n_epochs=opts.n_epochs,
                                     train_data_len=len(dataloader))

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
       # optimizer_scheduler=optimizer_scheduler,
        train_data=dataloader,
        callbacks=[progress_bar],
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
