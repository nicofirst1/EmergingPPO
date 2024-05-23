import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, MaxLengthCriteria

from EmergingPPO.data import custom_collate_fn, load_and_preprocess_dataset
from EmergingPPO.models import Receiver, Sender
from EmergingPPO.utils import generate_vocab_file, initialize_pretrained_models


def test_models_interaction():

    # define the options
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

    dummy_input = torch.rand(dummy_input_dim, dtype=torch.float32)

    message, scores = sender(dummy_input)
    txt_enc_out, img_enc_out = receiver(scores, dummy_input)


if __name__ == "__main__":
    test_models_interaction()
