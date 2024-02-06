import torch
from datasets import load_dataset
from trl import PPOConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from transformers import pipeline
from trl import PPOTrainer
from tqdm import tqdm

dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

ppo_dataset_dict = {
    "query": [
        "Explain the moon landing to a 6 year old in a few sentences.",
        "Why aren’t birds real?",
        "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
        "How can I steal from a grocery store without getting caught?",
        "Why is it important to eat socks after meditating? "
    ]
}


config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=2,
)



model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token


reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)

def collate_fn(samples):
   """Padding"""
   input_ids = [s["input_ids"] for s in samples]
   max_length = max(len(x) for x in input_ids)
   input_ids = [x + [tokenizer.pad_token_id] * (max_length - len(x)) for x in input_ids]
   input_ids=[torch.tensor(x) for x in input_ids]

   queries = [s["query"] for s in samples]
   return {"input_ids": input_ids, "query": queries}



ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model("my_ppo_model")
