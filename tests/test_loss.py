import copy

import torch
from transformers import BertTokenizerFast

from EmergingPPO.losses import NTXentLoss
from EmergingPPO.models import Receiver, Sender
from EmergingPPO.utils import generate_vocab_file


def test_loss():
    # define the batch size and the number of distractors
    batch_size = 2
    vocab_size = 10
    max_length = 6
    dummy_input_dim = [batch_size, 197, 768]

    # tokenizer
    vocab_file = generate_vocab_file(vocab_size)
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    tokenizer.bos_token_id = tokenizer.cls_token_id

    sender = Sender(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_length=max_length,
        gs_temperature=1,
    )

    receiver = Receiver(tokenizer, linear_dim=123, vocab_size=vocab_size)

    copy.deepcopy(list(sender.parameters()))
    copy.deepcopy(list(receiver.parameters()))

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
    dummy_input = torch.randn(dummy_input_dim)
    # forward pass through the sender
    message_logits, scores = sender(dummy_input)

    receiver_out = receiver(scores=scores, receiver_input=dummy_input)

    txt_enc_out, img_enc_out = receiver_out

    l, acc = loss.modified_ntxent_loss(txt_enc_out, img_enc_out, labels)

    l.backward()
    optimizer.step()

    print(message_logits)


if __name__ == "__main__":
    test_loss()
