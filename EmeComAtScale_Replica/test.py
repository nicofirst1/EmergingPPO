import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, ViTImageProcessor, ViTModel

from EmeComAtScale_Replica.losses import NTXentLoss
from EmeComAtScale_Replica.modeling import Sender, Receiver


def test():
    text_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
    img_checkpoint="google/vit-base-patch16-224-in21k"

    tokenizer = GPT2TokenizerFast.from_pretrained(text_checkpoint)

    image_processor = ViTImageProcessor.from_pretrained(img_checkpoint)

    img_encoder = ViTModel.from_pretrained(img_checkpoint)

    # freeze the encoder
    for param in img_encoder.parameters():
        param.requires_grad = False

    sender = Sender(img_encoder=img_encoder, tokenizer=tokenizer, image_processor=image_processor)
    receiver = Receiver(img_encoder=img_encoder, tokenizer=tokenizer, image_processor=image_processor)

    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    dataset = dataset.filter(lambda e, i: i < 100, with_indices=True)

    loss = NTXentLoss(
        temperature=1,
        similarity="cosine",
    )

    batch_size = 2
    distractors = 3

    sender_in = dataset[:batch_size]["image"]

    message_logits, scores = sender(sender_in)  # [bsz, seqlen, vocab_size]

    # Generate random indices
    indices = torch.randperm(len(dataset))

    # Select the batches
    batches = []
    for i in range(0, batch_size):
        batch_indices = indices[i:i + distractors].tolist()
        batch = [dataset[i]["image"] for i in batch_indices]
        batches.append(batch)

    receiver_out = receiver(scores=scores, receiver_input=batches)

    txt_enc_out, img_enc_out = receiver_out

    # repeat txt_enc_out for each distractor
    # txt_enc_out = txt_enc_out.unsqueeze(dim=1).repeat(1,distractors, 1)
    correct_img = torch.tensor([0, 1])
    l = loss.modified_ntxent_loss(txt_enc_out, img_enc_out, correct_img)

    print(message_logits)


if __name__ == "__main__":
    test()
