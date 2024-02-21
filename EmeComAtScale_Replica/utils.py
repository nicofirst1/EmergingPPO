import json
import os

from transformers import GPT2TokenizerFast, ViTImageProcessor, ViTModel

def initialize_pretrained_models(img_checkpoint="google/vit-base-patch16-224-in21k"):
    """
    Initialize an image processor, and an image encoder from pretrained models.
    The parameters of the image encoder are frozen.

    Args:
        img_checkpoint (str): The checkpoint for the image model.

    Returns:
        ViTImageProcessor: The initialized image processor.
        ViTModel: The initialized image encoder with frozen parameters.
    """
    image_processor = ViTImageProcessor.from_pretrained(img_checkpoint)
    img_encoder = ViTModel.from_pretrained(img_checkpoint)

    # freeze the encoder
    for param in img_encoder.parameters():
        param.requires_grad = False

    return image_processor, img_encoder



def generate_vocab_file(vocab_size, filename="vocab.txt"):
    """
    Generate a vocabulary file with a given size.

    Args:
        vocab_size (int): The size of the vocabulary.
        filename (str): The name of the file to save the vocabulary.
    """

    vocab_dict={k:k for k in range(vocab_size)}
    with open(filename, "w") as f:
        json.dump(vocab_dict, f)

    # get file path

    filename = os.path.abspath(filename)

    return filename