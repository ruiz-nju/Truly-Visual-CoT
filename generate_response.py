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
from utils.tools import read_json, save_json, print_info
from utils.logger import setup_logger
import time
from qwen2_5_methods import get_response_qwen2_5, refocus_qwen2_5
import pdb

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
        print(f"Loading {model_to_fullname[model_name]} on {DEVICE}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_to_fullname[model_name],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_to_fullname[model_name])
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor


def generation_mathvista_origin(model_name, data_dir, output_file_path):
    print(f"Saved results to {output_file_path}")
    model, processor = get_model(model_name)
    data_list = load_dataset("AI4Math/MathVista", split="testmini")
    data = {item["pid"]: item for item in data_list}
    query_file = os.path.join(data_dir, "query.json")
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

    for i, problem_id in enumerate(tqdm(test_pids, desc="Generating origin response")):
        problem: dict = data[problem_id].copy()
        # 加载的数据集中包含 decoded_image
        # 这里为了使用统一的函数，删除该键值对，直接根据路径加载本地图片
        problem.pop("decoded_image")
        image_path = os.path.join(data_dir, problem["image"])
        query = query_data[problem_id]
        try:
            if model_name == "qwen2_5":
                response = get_response_qwen2_5(
                    user_prompt=query,
                    image_path=image_path,
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


def generation_mathvista_refocus(
    model_name,
    output_file_path,
    input_file,
):
    model, processor = get_model(model_name)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    print(f"Loading {input_file}...")
    inputs = read_json(input_file)
    full_pids = list(inputs.keys())

    for i, pid in enumerate(tqdm(full_pids, desc="Generating refocus response")):

        problem = inputs[pid]
        query = problem["query"]
        image_path = os.path.join("data/mathvista", problem["image"])
        ori_response = problem["response"]

        if model_name == "qwen2_5":
            refocus_position, refocus_response = refocus_qwen2_5(
                model=model,
                processor=processor,
                user_prompt=query,
                image_path=image_path,
                ori_response=ori_response,
            )
            # 此处将原来的 response 替换为 refocus_response
            inputs[pid]["response"] = refocus_response
            inputs[pid]["refocus_position"] = refocus_position
            save_every = 5
            if (i % save_every == 0 and i > 0) or i == len(full_pids) - 1:
                try:
                    save_json(inputs, output_file_path)
                except Exception as e:
                    print(f"Error in saving {output_file_path}")
                    print(e)
        else:
            raise ValueError(f"Model {model_name} not supported")


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    period = args.period
    if period == "origin":
        output_dir = os.path.join("outputs_origin", dataset, model_name)
    elif period == "refocus":
        output_dir = os.path.join("outputs_refocus", dataset, model_name)
    else:
        raise ValueError(f"Period {period} not supported")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    setup_logger(os.path.join(output_dir, "logs", "generate_response_log.txt"))
    print_info(
        module_name=f"generate_response.py",
        model_name=model_name,
        dataset_name=dataset,
        period=period,
    )
    if dataset == "mathvista":
        output_file_path = os.path.join(output_dir, "generated_response.json")
        data_dir = os.path.join("data", "mathvista")
        if period == "origin":
            generation_mathvista_origin(
                model_name=model_name,
                data_dir=data_dir,
                output_file_path=output_file_path,
            )
        elif period == "refocus":
            input_file = os.path.join(
                "outputs_origin", dataset, model_name, "generated_response.json"
            )
            generation_mathvista_refocus(
                model_name=model_name,
                output_file_path=output_file_path,
                input_file=input_file,
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
    parser.add_argument(
        "--period",
        type=str,
        default="origin",
        choices=["origin", "refocus"],
    )
    args = parser.parse_args()
    main(args)
