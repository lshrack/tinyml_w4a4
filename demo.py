import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantize import quantize
from src.helpers import evaluate
from src.smooth import smooth_lm
from src.calibration import get_act_scales

OPT_MODEL_PATH = "facebook/opt-1.3b"
LLAMA_MODEL_PATH = "meta-llama/Llama-3.2-1B"

def get_input(options, value_name):
    while True:
        chosen_option = input(f'Please select a {value_name}. Options: {', '.join(options)}')
        if chosen_option in options:
            return chosen_option
        else:
            print("Invalid choice - please try again!")


def baseline_experiments(model_path, tokenizer, group_sizes=[128, 64, 32, 16]):
    print("Evaluating baseline group W4A4 quantization:")

    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except: pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        quantize(model, 4, weight_quant="per_group", act_quant="per_group", group_size=group_size)

        model_perplexity = evaluate(model, tokenizer)
        print(f"\nModel perplexity with group size {group_size}: {model_perplexity:.2f}")

def smoothquant(model_path, tokenizer, alpha, group_sizes=[128, 64, 32, 16]):
    print("Evaluating group W4A4 quantization with SmoothQuant:")
    model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
    act_scales = get_act_scales(model, tokenizer)
    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except: pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        smooth_lm(model, act_scales, alpha)
        quantize(model, 4, weight_quant="per_group", act_quant="per_group", group_size=group_size)

        model_perplexity = evaluate(model, tokenizer)
        print(f"\nModel perplexity with group size {group_size}: {model_perplexity:.2f}")

if __name__ == "__main__":
    model_type = get_input(['opt', 'llama'], "model")
    method_choice = get_input(['naive', 'smoothquant', 'awq', 'all'], "W4A4 quantization method")

    if model_type == "opt":
        model_path = OPT_MODEL_PATH
        alpha = 0.45
    elif model_type == "llama":
        model_path = LLAMA_MODEL_PATH
        alpha = 0.75
    else:
        raise Exception("unrecognized model type")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if method_choice in ('naive', 'all'):
        baseline_experiments(model_path, tokenizer)
    
    if method_choice in ('smoothquant', 'all'):
        smoothquant(model_path, tokenizer, alpha)
    
# todo 1 - combine quantize_opt and quantize_llama
# define model path in main func + pass
# get tokenizer in main func + pass