import sys
import torch
sys.path.append("..")

from collie import Server, LlamaForCausalLM, DashProvider, CollieConfig, env, MossForCausalLM, ChatGLMForCausalLM
from transformers import LlamaTokenizer, GenerationConfig, BitsAndBytesConfig

config = CollieConfig.from_pretrained("/mnt/petrelfs/zhangshuo/model/llama-7b-hf", trust_remote_code=True)
config.pp_size = 1
config.tp_size = 1
config.quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlamaForCausalLM.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf", config=config).cuda()
tokenizer = LlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/zhangshuo/model/llama-7b-hf", add_eos_token=False)
data_provider = DashProvider(tokenizer=tokenizer)
data_provider.generation_config = GenerationConfig(max_new_tokens=250)
server = Server(model, data_provider, config=config)
server.run()