from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, MaxLengthCriteria

from EmergingPPO.data import custom_collate_fn, load_and_preprocess_dataset
from EmergingPPO.models import Receiver, Sender
from EmergingPPO.utils import generate_vocab_file, initialize_pretrained_models


def test_models():

    # define the options
    vocab_size = 10  # must be higher than 4
    max_len = 4
    data_split = "train"
    vision_chk = "google/vit-base-patch16-224-in21k"
    data_subset = 8  # only load 8 data points
    gs_temperature = 1.0
    projection_output_dim = 4
    batch_size = 2

    img_encoder = initialize_pretrained_models(vision_chk)

    # tokenizer
    vocab_file = generate_vocab_file(vocab_size)
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    tokenizer.bos_token_id = tokenizer.cls_token_id

    # stopping criteria as maxlength for decoder
    stopping_criteria = MaxLengthCriteria(max_length=max_len)

    sender = Sender(
        img_encoder=img_encoder,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        stopping_criteria=stopping_criteria,
        gs_temperature=gs_temperature,
    )
    receiver = Receiver(
        img_encoder=img_encoder,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        linear_dim=projection_output_dim,
    )

    dataset = load_and_preprocess_dataset(
        "Maysee/tiny-imagenet",
        data_split,
        vision_chk,
        data_subset=data_subset,
    )

    train_data = dataset[0]

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    for batch in train_dataloader:
        sender_input, labels, receiver_input, aux_input = batch
        message, scores = sender(sender_input)
        txt_enc_out, img_enc_out = receiver(scores, receiver_input)

        break


if __name__ == "__main__":
    test_models()
