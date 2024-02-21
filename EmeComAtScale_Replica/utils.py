from transformers import GPT2TokenizerFast, ViTImageProcessor, ViTModel

def initialize_pretrained_models(text_checkpoint="nlpconnect/vit-gpt2-image-captioning", img_checkpoint="google/vit-base-patch16-224-in21k"):
    """
    Initialize a tokenizer, an image processor, and an image encoder from pretrained models.
    The parameters of the image encoder are frozen.

    Args:
        text_checkpoint (str): The checkpoint for the text model.
        img_checkpoint (str): The checkpoint for the image model.

    Returns:
        GPT2TokenizerFast: The initialized tokenizer.
        ViTImageProcessor: The initialized image processor.
        ViTModel: The initialized image encoder with frozen parameters.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(text_checkpoint)
    image_processor = ViTImageProcessor.from_pretrained(img_checkpoint)
    img_encoder = ViTModel.from_pretrained(img_checkpoint)

    # freeze the encoder
    for param in img_encoder.parameters():
        param.requires_grad = False

    return tokenizer, image_processor, img_encoder