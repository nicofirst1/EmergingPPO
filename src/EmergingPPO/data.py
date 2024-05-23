from typing import List

import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyBatch
from joblib import Memory
from transformers import ViTImageProcessor

from EmergingPPO.utils import initialize_pretrained_models

location = "./cachedir"
memory = Memory(location, verbose=1)

from typing import Optional, Union


##########################################################
### Keep arguments simple such that caching works fine ###
##########################################################
@memory.cache
def load_and_preprocess_dataset(
    dataset_key: str,
    split: str,
    vision_chk: str,
    data_subset: Optional[Union[float, int]] = None,
    load_from_cache_file: bool = False,
):

    # if split is all, load both train and test
    if split == "all":
        split = ["train", "valid"]
    else:
        split = [split]

    # if data_subset is not 1.0, load only a subset of the data
    if data_subset is not None:

        if isinstance(data_subset, int):
            # If int, we treat as absolute number
            split = [f"{s}[:{data_subset}]" for s in split]
        elif isinstance(data_subset, float):
            # If float, we treat as percentage
            assert (
                data_subset < 1.0
            ), "float values for data_subset must be in ]0.0,1.0[ and are treated as percentage"
            split = [f"{s}[:{int(data_subset * 100)}%]" for s in split]
        else:
            raise ValueError(
                "data_subset should be either absolute int, or relative float in ]0.0,1.0[. Use data_subset=None to load all data."
            )

    print("Splits just before loading:", split)
    dataset = [load_dataset(dataset_key, split=s) for s in split]
    print("List of datasets right after loading:", dataset)

    # when loading two splits the dataset is a list, if not then only one split is loaded
    # not needed anymore b/c of list comprehension above
    # if not isinstance(dataset, list):
    #    dataset = list(dataset)

    # filter all images where the mode is not RBG
    dataset = [d.filter(lambda e: e["image"].mode == "RGB") for d in dataset]

    image_processor = ViTImageProcessor.from_pretrained(vision_chk)
    image_encoder = initialize_pretrained_models(vision_chk)

    # concat processor and encoder
    image_fn = lambda kwargs: image_encoder(
        image_processor(**kwargs).data["pixel_values"]
    )

    # preprocess the images
    dataset = [
        d.map(
            data_map,
            batched=True,
            with_indices=True,
            remove_columns=["image"],
            fn_kwargs={
                "image_processor": image_fn,
            },
            num_proc=1,
            load_from_cache_file=load_from_cache_file,
        )
        for d in dataset
    ]
    return dataset


def data_map(
    example: LazyBatch,
    indices: List[int],
    image_processor,
):

    images_list = example["image"]

    # process every image with image processor and then embed it with the image encoder
    # dim [batch, 197, 768]
    # pooled is [batch, 768]
    image = image_processor(dict(images=images_list, return_tensors="pt"))
    image = image.last_hidden_state

    batch_size = len(image)
    labels = []

    labels = torch.zeros((batch_size, 1), dtype=torch.int)
    # labels = images_list

    res_dict = {"sample": image, "label": labels, "img_id": indices}

    return res_dict


def custom_collate_fn(batch):
    receiver_input = torch.stack([torch.as_tensor(item["sample"]) for item in batch])
    # dim [batch_size, 1,197, 768]
    labels = torch.stack([torch.as_tensor(item["label"]) for item in batch])
    img_ids = torch.as_tensor([item["img_id"] for item in batch])
    sender_input = receiver_input

    return sender_input, labels, receiver_input, img_ids
