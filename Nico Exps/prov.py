from transformers import T5Config, BertTokenizer, GenerationConfig

from agents import CustomT5

tokenizer = BertTokenizer(
    "/Users/giulia/Desktop/EmergingPPO/Nico Exps/variable_compositionality/data/two_predicates/preprocess/0.01_train_mcd/dec_preprocess/vocab.txt")


tokenizer.add_special_tokens({"eos_token": "[EOS]"})

model_configs = T5Config(
    vocab_size=tokenizer.vocab_size+len(tokenizer.all_special_ids),
)
model = CustomT5(model_configs)

input_ids = tokenizer(
    "Melanie scares Alistair", return_tensors="pt"
).input_ids  # Batch size 1

gen_config=GenerationConfig(
bos_token_id=tokenizer.cls_token_id,
pad_token_id=tokenizer.pad_token_id,
eos_token_id=tokenizer.eos_token_id,

)


gen = model.generate(input_ids, generation_config=gen_config)
# forward pass
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state
