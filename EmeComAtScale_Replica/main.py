from functools import partial

import torch
from datasets import load_dataset
from egg.core import Trainer, ProgressBarLogger
from egg.zoo.emcom_as_ssl.utils import get_common_opts
from transformers import GPT2TokenizerFast, ViTImageProcessor, ViTModel

from EmeComAtScale_Replica.data import emecom_map, custom_collate_fn
from EmeComAtScale_Replica.losses import NTXentLoss
from modeling import Sender, Receiver, EmComSSLSymbolGame




def main(args):
    opts = get_common_opts(params=args)
    print(f"{opts}\n")

    text_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
    img_checkpoint = "google/vit-base-patch16-224-in21k"

    tokenizer = GPT2TokenizerFast.from_pretrained(text_checkpoint)

    image_processor = ViTImageProcessor.from_pretrained(img_checkpoint)

    img_encoder = ViTModel.from_pretrained(img_checkpoint)

    sender = Sender(img_encoder=img_encoder,tokenizer=tokenizer, image_processor=image_processor)
    receiver = Receiver(img_encoder=img_encoder,tokenizer=tokenizer, image_processor=image_processor)

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
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5
    )

    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    # todo: remove after testing
    #dataset = dataset.filter(lambda e, i: i < 100, with_indices=True)

    # filter all images where the mode is not RBG
    dataset = dataset.filter(lambda e: e["image"].mode == "RGB")

    dataset = dataset.map(emecom_map, batched=True, remove_columns=["image"],
                          fn_kwargs={"num_distractors": 3, "image_processor": image_processor},
                          #load_from_cache_file=False
                          )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    epochs=10

    progress_bar=ProgressBarLogger(n_epochs=epochs,
                                   train_data_len=len(dataloader))

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=dataloader,
        callbacks=[progress_bar],
    )
    trainer.train(n_epochs=epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])


