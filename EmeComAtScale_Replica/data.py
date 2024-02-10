import random

import torch


def emecom_map(example, num_distractors, image_processor):



    # process every image
    image = image_processor(example['image'], return_tensors="pt").data['pixel_values']
    #image =example['image']
    samples=[]
    labels=[]

    batch_size = len(image)

    for idx in range(batch_size):
        target = image[idx]

        # get distractors random indices from batch_size excluded the idx
        indices = list(range(batch_size))
        indices.remove(idx)
        indices = random.sample(indices, num_distractors)


        distractors = [image[i] for i in indices]
        sample= [target]+ distractors

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
    receiver_input =torch.stack([torch.as_tensor(item['sample']) for item in batch])
    labels = torch.stack([torch.as_tensor(item['label']) for item in batch])

    sender_input = receiver_input[torch.arange(receiver_input.shape[0]), labels]

    return sender_input, labels, receiver_input