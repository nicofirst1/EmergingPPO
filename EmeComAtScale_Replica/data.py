import random
from argparse import Namespace
from typing import List

import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyBatch
from joblib import Memory
from transformers import ViTImageProcessor

location = "./cachedir"
memory = Memory(location, verbose=0)

from typing import Optional, Union


@memory.cache
def load_and_preprocess_dataset(dataset_key:str,
                                split:str,
                                vision_chk:str,
                                distractors_num:Optional[int]=0,  # Legacy, we use batch size instead
                                data_subset:Optional[Union[float,int]]=None):

    # if split is all, load both train and test
    if split == "all":
        split = ["train", "valid"]
    else:
        split = [split]

    # if data_subset is not 1.0, load only a subset of the data
    if data_subset is not None:
        # if it i less than zero we need to take a percentage of the data
        if data_subset < 0:
            split = [f"{s}[:{int(data_subset * 100)}%]" for s in split]
        else:
            split = [f"{s}[:{int(data_subset)}]" for s in split]

    # todo: load all splits
    dataset = load_dataset(dataset_key, split=split)

    # when loading two splits the dataset is a list, if not then only one split is loaded
    if not isinstance(dataset, list):
        dataset = list(dataset)

    # filter all images where the mode is not RBG
    dataset = [d.filter(lambda e: e["image"].mode == "RGB") for d in dataset]

    image_processor = ViTImageProcessor.from_pretrained(vision_chk)

    # preprocess the images
    dataset = [
        d.map(
            emecom_map,
            batched=True,
            with_indices=True,
            remove_columns=["image"],
            fn_kwargs={
                "num_distractors": distractors_num,
                "image_processor": image_processor,
            },
            num_proc=1,
            load_from_cache_file=False,
        )
        for d in dataset
    ]
    return dataset


def emecom_map(
    example: LazyBatch,
    indices: List[int],
    num_distractors: int,
    image_processor: ViTImageProcessor,
):

    images_list = example["image"]

    # process every image
    image = image_processor(images_list, return_tensors="pt").data["pixel_values"]

    batch_size = len(image)
    samples = []
    labels = []

    if num_distractors < 1:
        samples = image
        labels = torch.zeros((batch_size, 1), dtype=torch.int)
        # labels = images_list

    else:

        for idx in range(batch_size):
            target = image[idx]

            # get distractors random indices from batch_size excluded the idx
            indices = list(range(batch_size))
            indices.remove(idx)
            indices = random.sample(indices, num_distractors)

            distractors = [image[i] for i in indices]
            sample = [target] + distractors

            # randomly permute the samples and save the target index
            indices = torch.randperm(len(sample))
            sample = [sample[i] for i in indices]
            samples.append(sample)
            labels.append(indices[0])

    res_dict = {"sample": samples, "label": labels, "img_id": indices}

    return res_dict


def custom_collate_fn(batch):
    receiver_input = torch.stack([torch.as_tensor(item["sample"]) for item in batch])
    labels = torch.stack([torch.as_tensor(item["label"]) for item in batch])
    img_ids = torch.as_tensor([item["img_id"] for item in batch])
    sender_input = receiver_input[torch.arange(receiver_input.shape[0]), labels]

    return sender_input, labels, receiver_input, img_ids
