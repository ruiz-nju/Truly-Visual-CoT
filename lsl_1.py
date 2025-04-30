import pdb
import os
import argparse
from ruamel.yaml import YAML

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from utils.tools import encode_base64

from utils.tools import read_json, save_json, print_info
from utils.logger import setup_logger

import os
import argparse
from PIL import Image
from ruamel.yaml import YAML
from tqdm import tqdm

from models.Qwen import Qwen2_5_VLForInterCoT
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

IMG_FOLDER = None  # TODO
EVAL_FILE = None  # TODO
DATA_NAME = "m3cot"


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
        print(f"Loading {model_to_fullname[model_name]} on {DEVICE}")
        model_path = "/mnt/hdd/zhurui/models/Qwen2.5-VL-7B-Instruct"  # TODO:修改路径
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForInterCoT.from_pretrained(
            model_path, torch_dtype="auto", attn_implementation="eager"
        ).to("cuda")
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor


# TODO：是否需要设计一下Prompt
def prepare_qwen2_5_input(
    user_prompt, image_path, processor, cur_generation=None, return_image_masks=False
):
    """
    Prepare the input for Qwen2.5VL.
    """
    image = Image.open(image_path)
    image_str = encode_base64(image=image)
    user_message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    image_inputs, video_inputs = process_vision_info(user_message)
    text = processor.apply_chat_template(
        user_message, tokenize=False, add_generation_prompt=True
    )
    if cur_generation:
        text = text + cur_generation
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs, sub_image_masks= processor(text = prompt, images=vision_x, padding=True, return_tensors="pt", return_for_text_completion=False, return_image_masks=True)

    inputs = inputs.to(device="cuda", dtype=torch.bfloat16)
    inputs["output_attentions"] = True

    return inputs


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def get_response_qwen2_5(
    user_prompt, image_path, model, processor, cur_generation=None
):
    generation_config = {
        "do_sample": False,
        "max_new_tokens": 5096,
        # "temperature": 0.7,
        # "top_p": 0.9,
        # "repetition_penalty": 1.2,
        # "min_new_tokens": 32,
        # "max_new_tokens": 2048,
    }  # max_new_tokens=5096
    inputs = prepare_qwen2_5_input(
        user_prompt=user_prompt,
        image_path=image_path,
        processor=processor,
        cur_generation=cur_generation,
    ).to(DEVICE, torch.bfloat16)
    generate_ids = model.generate(**inputs, **generation_config)
    generation = processor.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    return generation.split("\nassistant\n")[-1].strip()


def generation_mathvista_origin(model_name, data_dir, output_file_path):

    print(f"Saving results to {output_file_path}")
    model, processor = get_model(model_name)
    # tokens = processor.tokenizer.tokenize("Wait, let me double-check the image.")
    # pids = processor.tokenizer.convert_tokens_to_ids(tokens)
    # pdb.set_trace()
    data_file = os.path.join(data_dir, "testmini.json")
    if not os.path.exists(data_file):
        raise ValueError(f"Data file {data_file} not found")
    data = read_json(data_file)
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
        problem = data[problem_id]
        image_path = os.path.join(data_dir, problem["image"])
        query = query_data[problem_id]

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
        save_every = 1
        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
            except Exception as e:
                print(f"Error in saving {output_file_path}")
                print(e)

    print("MathVista: Generating Responses - Finish")


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
        generation_mathvista_origin(
            model_name=model_name,
            data_dir=data_dir,
            output_file_path=output_file_path,
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
