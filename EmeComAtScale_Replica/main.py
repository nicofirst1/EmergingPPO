from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login

from modeling import ViTGPT2Agent



def interaction(sender_agent, receiver_agent, dataset, num_steps):
    pass


def main(args):
    sender = ViTGPT2Agent()
    dataset = load_dataset("imagenet-1k")


    dummy_example = dataset[0]["image"]

    message_logits = sender(dummy_example)  # [bsz, seqlen, vocab_size]

    print(message_logits)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
