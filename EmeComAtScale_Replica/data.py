import random

import torch
from datasets import load_dataset
from joblib import Memory
from transformers import ViTImageProcessor

location = './cachedir'
memory = Memory(location, verbose=0)


@memory.cache
def load_and_preprocess_dataset(dataset_key, split, image_processor_key, num_distractors=0):
    # todo: load all splits
    dataset = load_dataset(dataset_key, split=split)

    # when loading two splits the dataset is a list, if not then only one split is loaded
    if not isinstance(dataset, list):
        dataset = list(dataset)

    # filter all images where the mode is not RBG
    dataset = [d.filter(lambda e: e["image"].mode == "RGB") for d in dataset]

    image_processor = ViTImageProcessor.from_pretrained(image_processor_key)

    # preprocess the images
    dataset = [d.map(emecom_map, batched=True, remove_columns=["image"],
                     fn_kwargs={"num_distractors": num_distractors, "image_processor": image_processor},
                     num_proc=1) for d in dataset]
    return dataset


def emecom_map(example, num_distractors: int, image_processor: ViTImageProcessor):
    # process every image
    image = image_processor(example['image'], return_tensors="pt").data['pixel_values']

    batch_size = len(image)
    samples = []
    labels = []

    if num_distractors < 1:
        samples = image
        labels = torch.zeros((batch_size, 1), dtype=torch.int)

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

    res_dict = {
        "sample": samples,
        "label": labels
    }

    return res_dict


def custom_collate_fn(batch):
    receiver_input = torch.stack([torch.as_tensor(item['sample']) for item in batch])
    labels = torch.stack([torch.as_tensor(item['label']) for item in batch])

    sender_input = receiver_input[torch.arange(receiver_input.shape[0]), labels]

    return sender_input, labels, receiver_input
