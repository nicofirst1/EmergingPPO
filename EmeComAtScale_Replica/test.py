import copy

import torch
from datasets import load_dataset

from EmeComAtScale_Replica.data import emecom_map, custom_collate_fn
from EmeComAtScale_Replica.losses import NTXentLoss
from EmeComAtScale_Replica.models import Sender, Receiver
from EmeComAtScale_Replica.utils import initialize_pretrained_models


def test():
    # define the batch size and the number of distractors
    batch_size = 2
    distractors = 3

    # initialize the  models
    tokenizer, image_processor, img_encoder = initialize_pretrained_models()

    sender = Sender(img_encoder=img_encoder, tokenizer=tokenizer, image_processor=image_processor)
    receiver = Receiver(img_encoder=img_encoder, tokenizer=tokenizer, image_processor=image_processor)


    sender_w=copy.deepcopy(list(sender.parameters()))
    receiver_w=copy.deepcopy(list(receiver.parameters()))
    img_encoder_w=copy.deepcopy(list(img_encoder.parameters()))

    # load the dataset
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    dataset = dataset.filter(lambda e, i: i < 100, with_indices=True)
    # preprocess the images
    dataset = dataset.map(emecom_map, batched=True, remove_columns=["image"],
                          fn_kwargs={"num_distractors": distractors, "image_processor": image_processor},)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    # loss and optimizer
    loss = NTXentLoss(
        temperature=1,
        similarity="cosine",
    )

    model_parameters = list(sender.parameters()) + list(receiver.parameters())

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=1,
        momentum=0.9,
    )

    ################################################################################
    # START OF BATCH PROCESSING
    ################################################################################

    optimizer.zero_grad()
    # select the batch
    batch =dataloader.__iter__().__next__()
    sender_input, labels, receiver_input = batch
    # forward pass through the sender
    message_logits, scores = sender(sender_input)  # [bsz, seqlen, vocab_size]


    receiver_out = receiver(scores=scores, receiver_input=receiver_input)

    txt_enc_out, img_enc_out = receiver_out

    l, acc = loss.modified_ntxent_loss(txt_enc_out, img_enc_out, labels)

    l.backward()
    optimizer.step()



    print(message_logits)


if __name__ == "__main__":
    test()
