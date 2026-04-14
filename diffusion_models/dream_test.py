import torch
from transformers import AutoModel, AutoTokenizer
import os

model_id = "Dream-org/Dream-v0-Instruct-7B"
cache_dir = os.getenv("HF_HUB_CACHE", None)

if cache_dir is None:
    print("couldn't locacte HF CACHE")
    exit()

model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [
    {"role": "user", "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=True,
    steps=512,
    temperature=0.2,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]


print(f"Lenght of Generations = {len(generations)}")
print("\n")
print(generations)
print("\n")
print(generations[0].split(tokenizer.eos_token)[0])
