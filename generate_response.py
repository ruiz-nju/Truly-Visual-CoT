import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import argparse
from utils.tools import read_json, save_json
from utils.logger import setup_logger
import time
from qwen2_5_methods import get_response_qwen2_5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
}

dataset_to_full_name = {"mathvista": "AI4Math/MathVista"}


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def get_model(model_name):
    if model_name == "qwen2_5":
        # max_pixels = 256 * 28 * 28
        print(f"Loading {model_to_fullname[model_name]} on {DEVICE}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_to_fullname[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_to_fullname[model_name])
        # processor = AutoProcessor.from_pretrained(
        #     model_to_fullname[model_name], max_pixels=max_pixels
        # )
        # processor.image_processor.size["longest_edge"] = (
        #     max_pixels  # this is likely a bug in current transformers (4.50.0) library, passing in max_pixels to from_pretrained does not work
        # )
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor


def generation_mathvista(model_name, output_file_path):
    print(f"Saved results to {output_file_path}")
    model, processor = get_model(model_name)
    data_list = load_dataset("AI4Math/MathVista", split="testmini")
    data = {item["pid"]: item for item in data_list}
    query_file = "./data/mathvista/query.json"
    if os.path.exists(query_file):
        print(f"Loading existing {query_file}...")
        query_data = read_json(query_file)
    else:
        raise ValueError(f"Query file {query_file} not found")
    full_pids = list(data.keys())
    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_pids = []
    for problem_id in full_pids:
        if problem_id in results and "response" in results[problem_id]:
            response = results[problem_id]["response"]
            if verify_response(response):
                skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    for i, problem_id in enumerate(tqdm(test_pids, desc="Genrating response")):
        problem: dict = data[problem_id].copy()

        # Remove decoded Image for JSON deserialization
        problem_decoded_image = problem["decoded_image"]
        problem.pop("decoded_image")

        query = query_data[problem_id]
        try:
            if model_name == "qwen2_5":
                response = get_response_qwen2_5(
                    user_prompt=query,
                    decoded_image=problem_decoded_image,
                    model=model,
                    processor=processor,
                )
            else:
                raise ValueError(f"Model {model_name} not supported")
            results[problem_id] = problem
            results[problem_id]["query"] = query
            results[problem_id]["response"] = response
        except Exception as e:
            print(f"Error in extracting answer for {problem_id}")
            print(e)
            results[problem_id] = problem
            results[problem_id]["error"] = str(e)

        save_every = 5
        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
            except Exception as e:
                print(f"Error in saving {output_file_path}")
                print(e)

    print("MathVista: Generating Responses - Finish")


def print_info(model_name, dataset_name):
    print(
        f"""
    "================================================================"
    "🚀 Running generate_response.py"
    "📦 Model:   {model_name}"
    "📚 Dataset: {dataset_name}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    output_dir = os.path.join("outputs", dataset, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    setup_logger(os.path.join(output_dir, "logs", "generate_response_log.txt"))
    print_info(model_name, dataset)
    if dataset == "mathvista":
        generation_mathvista(
            model_name, os.path.join(output_dir, "generated_response.json")
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
