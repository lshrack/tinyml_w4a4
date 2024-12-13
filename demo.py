import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantize import quantize_opt, quantize_llama
from src.helpers import evaluate


OPT_MODEL_PATH = "facebook/opt-1.3b"
LLAMA_MODEL_PATH = "meta-llama/Llama-3.2-1B"

def get_input(options, value_name):
    while True:
        chosen_option = input(f'Please select a {value_name}. Options: {', '.join(options)}')
        if chosen_option in options:
            return chosen_option
        else:
            print("Invalid choice - please try again!")


def baseline_experiments(model_type="opt", group_sizes=[128, 64, 32, 16]):
    if model_type == "opt":
        model_path = OPT_MODEL_PATH
    elif model_type == "llama":
        model_path = LLAMA_MODEL_PATH
    else:
        raise Exception("unrecognized model type")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except: pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        if model_type == 'opt':
            quantize_opt(model, 4, weight_quant="per_group", act_quant="per_group", group_size=group_size)
        elif model_type == 'llama':
            quantize_llama(model, 4, weight_quant="per_group", act_quant="per_group", group_size=group_size)
        else:
            raise Exception("unrecognized model type")
    
        model_perplexity = evaluate(model, tokenizer)
        print(f"\nModel perplexity with group size {group_size}: {model_perplexity:.2f}")

if __name__ == "__main__":
    model_choice = get_input(['opt', 'llama'], "model")
    method_choice = get_input(['naive', 'smoothquant', 'awq', 'all'], "W4A4 quantization method")

    if method_choice in ('naive', 'all'):
        baseline_experiments(model_choice)
    
