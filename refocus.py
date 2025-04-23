import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
import re
import os
from tqdm import tqdm
import argparse
from utils.tools import read_json
from utils.logger import setup_logger
import time
from qwen2_5_methods import refocus_qwen2_5, get_response_qwen2_5, prepare_qwen2_5_input

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
}

dataset_to_full_name = {"mathvista": "AI4Math/MathVista"}


def get_model(model_name):
    if model_name == "qwen2_5":
        max_pixels = 256 * 28 * 28
        print(f"Loading {model_to_fullname[model_name]} on {DEVICE}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_to_fullname[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            model_to_fullname[model_name], max_pixels=max_pixels
        )
        processor.image_processor.size["longest_edge"] = (
            max_pixels  # this is likely a bug in current transformers (4.50.0) library, passing in max_pixels to from_pretrained does not work
        )
    elif model_name == "qwen2":
        print(f"Loading {model_to_fullname[model_name]} on {DEVICE}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_to_fullname[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            model_to_fullname[model_name], max_pixels=max_pixels
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor


def print_info(model_name, dataset_name):
    print(
        f"""
    "================================================================"
    "🚀 Running refocus.py"
    "📦 Model:   {model_name}"
    "📚 Dataset: {dataset_name}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )


def refocus_mathvista(
    model_name,
    model,
    processor,
    input_file,
    output_dir,
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    print(f"Loading {input_file}...")
    inputs = read_json(input_file)
    full_pids = list(inputs.keys())

    for i, pid in enumerate(tqdm(full_pids, desc="Processing")):

        problem = inputs[pid]
        query = problem["query"]
        image_path = os.path.join("data/mathvista", problem["image"])
        ori_response = problem["response"]

        if model_name == "qwen2_5":
            refocus_qwen2_5(
                model=model,
                processor=processor,
                user_prompt=query,
                image_path=image_path,
                ori_response=ori_response,
            )
        else:
            raise ValueError(f"Model {model_name} not supported")


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    output_dir = os.path.join("outputs", dataset, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    setup_logger(os.path.join(output_dir, "logs", "refocus_log.txt"))
    print_info(model_name, dataset)
    model, processor = get_model(model_name)
    if dataset == "mathvista":
        input_file = os.path.join(
            "outputs_origin", dataset, model_name, "generated_response.json"
        )
        refocus_mathvista(
            model_name=model_name,
            model=model,
            processor=processor,
            input_file=input_file,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="qwen2_5", choices=model_to_fullname.keys()
    )
    parser.add_argument(
        "--dataset", type=str, default="mathvista", choices=dataset_to_full_name.keys()
    )
    args = parser.parse_args()
    main(args)
