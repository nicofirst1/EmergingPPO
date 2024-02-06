import torch
from datasets import load_dataset
from egg.core import Trainer
from egg.zoo.emcom_as_ssl.archs import EmComSSLSymbolGame
from egg.zoo.emcom_as_ssl.losses import NTXentLoss
from egg.zoo.emcom_as_ssl.utils import get_common_opts
from transformers import GPT2TokenizerFast, ViTImageProcessor

from modeling import Sender, Receiver


def test(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    image_processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    sender = Sender(tokenizer=tokenizer, image_processor=image_processor)
    receiver = Receiver(tokenizer=tokenizer, image_processor=image_processor)
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')

    batch_size = 2
    distractors = 3

    sender_in = dataset[:batch_size]["image"]

    message_logits = sender(sender_in)  # [bsz, seqlen, vocab_size]

    # Generate random indices
    indices = torch.randperm(len(dataset))

    # Select the batches
    batches = []
    for i in range(0, batch_size):
        batch_indices = indices[i:i + distractors].tolist()
        batch = [dataset[i]["image"] for i in batch_indices]
        batches.append(batch)

    receiver_out = receiver(message=message_logits, receiver_input=batches)

    print(message_logits)


def main(args):
    opts = get_common_opts(params=args)
    print(f"{opts}\n")

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    image_processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    sender = Sender(tokenizer=tokenizer, image_processor=image_processor)
    receiver = Receiver(tokenizer=tokenizer, image_processor=image_processor)

    loss = NTXentLoss(
        temperature=1,
        similarity="cosine",
        use_distributed_negatives=False,
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

    # todo add collate_fn (maybe preprocess before?)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=dataloader,
        callbacks=[],
    )
    trainer.train(n_epochs=5)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
