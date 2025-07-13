
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


#models--TheBloke--Mistral-7B-Instruct-v0.1-GPTQ      models--mistralai--Mixtral-8x7B-Instruct-v0.1
#models--bert-base-uncased                            models--mlx-community--Llama-3.2-3B-Instruct-4bit
#models--mistralai--Mistral-7B-Instruct-v0.1          models--mlx-community--Meta-Llama-3-8B-Instruct-4bit
#models--mistralai--Mistral-7B-Instruct-v0.2          models--mlx-community--quantized-gemma-7b
#models--mistralai--Mistral-7B-v0.1                   version.txt
#-n4 gpu
#RAG systems


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
cache_dir = "/Users/jordanharris/.cache/huggingface/hub"
local_model_path = "/Users/jordanharris/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c"




tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    #model_id,
    #token=token,
    #cache_dir=cache_dir,
    use_fast=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    #model_id,
    #cache_dir=cache_dir,
    #token=token,
    trust_remote_code=True,
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map = "auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    low_cpu_mem_usage=True,
    tokenizer=tokenizer,
)

response = generator(
    "You are a helpful AI. Return a JSON list of 3 fruits.",
    max_new_tokens=100,
    do_sample = False
)

print(response[0]["generated_text"])
