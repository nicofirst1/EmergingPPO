from torch.utils.data import DataLoader

from EmergingPPO.data import custom_collate_fn, load_and_preprocess_dataset


def test_data_processing():

    # define the options
    data_split = "all"
    vision_chk = "google/vit-base-patch16-224-in21k"
    data_subset = 8  # only load X data points
    batch_size = 2

    datasets = load_and_preprocess_dataset(
        "Maysee/tiny-imagenet",
        data_split,
        vision_chk,
        data_subset=data_subset,
        load_from_cache_file=False,
    )

    train_data = datasets[0]
    test_data = datasets[1]

    # check if the data is loaded correctly
    assert len(train_data) == data_subset
    assert len(test_data) == data_subset

    data_point = train_data[0]

    assert "sample" in data_point.keys()
    assert "label" in data_point.keys()

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    for batch in train_dataloader:
        sender_input, labels, receiver_input, aux_input = batch
        break

    print("Data processing test passed!")
