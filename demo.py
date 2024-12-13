import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantize import quantize
from src.helpers import evaluate
from src.smooth import smooth_lm, smooth_lm_two_alphas
from src.calibration import get_act_scales, get_calib_feat
from src.awq import awq_scale_model

OPT_MODEL_PATH = "facebook/opt-1.3b"
LLAMA_MODEL_PATH = "meta-llama/Llama-3.2-1B"


def get_input(options, value_name):
    while True:
        chosen_option = input(
            f'Please select a {value_name}. Options: {", ".join(options)}\n'
        )
        if chosen_option in options:
            return chosen_option
        else:
            print("Invalid choice - please try again!")


def baseline_experiments(model_path, tokenizer, group_sizes=[128, 64, 32, 16]):
    result = "baseline group W4A4 quantization:"
    print(f"Evaluating {result}")

    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        quantize(
            model,
            4,
            weight_quant="per_group",
            act_quant="per_group",
            group_size=group_size,
        )

        model_perplexity = evaluate(model, tokenizer)
        group_size_result = f"group size {group_size}: {model_perplexity:.2f}"
        result += f"\n{group_size_result}"
        print(f"\nBaseline model perplexity with {group_size_result}")

    return result + "\n"


def smoothquant(
    model_path, tokenizer, alpha, group_sizes=[128, 64, 32, 16], attn_only=False
):
    method = (
        "SmoothQuant, smoothing attention layers only" if attn_only else "SmoothQuant"
    )
    result = f"group W4A4 quantization with {method}:"
    print(f"Evaluating {result}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    act_scales = get_act_scales(model, tokenizer)
    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        if attn_only:
            smooth_lm_two_alphas(model, act_scales, alpha_qkv=alpha, alpha_fc=-1)
        else:
            smooth_lm(model, act_scales, alpha)

        quantize(
            model,
            4,
            weight_quant="per_group",
            act_quant="per_group",
            group_size=group_size,
        )

        model_perplexity = evaluate(model, tokenizer)

        group_size_result = f"group size {group_size}: {model_perplexity:.2f}"
        result += f"\n{group_size_result}"
        print(f"\nModel perplexity with SmoothQuant and {group_size_result}")

    del act_scales
    gc.collect()
    torch.cuda.empty_cache()

    return result + "\n"


def awq(model_path, tokenizer, group_sizes=[128, 64, 32, 16], act_quant="per_group"):
    a_bit = 16 if act_quant == "no_act_quant" else 4
    result = f"group W4A{a_bit} quantization with AWQ:"
    print(f"Evaluating {result}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    input_feat = get_calib_feat(model, tokenizer)

    for group_size in group_sizes:
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        awq_scale_model(model, input_feat, 4, group_size)
        quantize(
            model,
            4,
            weight_quant="per_group",
            act_quant=act_quant,
            group_size=group_size,
        )

        group_size_result = f"group size {group_size}: {model_perplexity:.2f}"
        result += f"\n{group_size_result}"

        model_perplexity = evaluate(model, tokenizer)
        print(f"\nModel perplexity with AWQ and {group_size_result}")

    del input_feat
    gc.collect()
    torch.cuda.empty_cache()

    return result + "\n"


if __name__ == "__main__":
    model_type = get_input(["opt", "llama"], "model")
    method_choice = get_input(
        [
            "naive",
            "smoothquant_w4a4",
            "smoothquant_w4a4_attn_only",
            "awq_w4a4",
            "awq_w4a16",
            "all",
        ],
        "W4A4 quantization method",
    )
    group_size = get_input(["128", "64", "32", "16", "all"], "group size")

    if model_type == "opt":
        model_path = OPT_MODEL_PATH
        alpha = 0.45
    elif model_type == "llama":
        model_path = LLAMA_MODEL_PATH
        alpha = 0.75
    else:
        raise Exception("unrecognized model type")

    if group_size == "all":
        group_sizes = [128, 64, 32, 16]
    else:
        group_sizes = [int(group_size)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    results = "RESULTS:\n\n"

    if method_choice in ("naive", "all"):
        results += baseline_experiments(model_path, tokenizer, group_sizes=group_sizes)

    if method_choice in ("smoothquant_w4a4", "all"):
        results += smoothquant(model_path, tokenizer, alpha, group_sizes=group_sizes)

    if method_choice in ("smoothquant_w4a4_attn_only", "all"):
        results += smoothquant(
            model_path, tokenizer, alpha, group_sizes=group_sizes, attn_only=True
        )

    if method_choice in ("awq_w4a4", "all"):
        results += awq(model_path, tokenizer, group_sizes=group_sizes)

    if method_choice in ("awq_w4a16", "all"):
        results += awq(
            model_path, tokenizer, group_sizes=group_sizes, act_quant="no_act_quant"
        )

    print(results)
