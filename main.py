from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Meta LLaMA model and tokenizer
llm_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the actual path or model name
tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForCausalLM.from_pretrained(llm_name)