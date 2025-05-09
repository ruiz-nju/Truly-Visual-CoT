import os
import pdb
import json
import torch
import argparse
import numpy as np


from tqdm import tqdm
from utils.tools import (
    read_json,
    save_json,
    print_info,
    split_sentence,
    DATASET_TO_FULL_NAME,
    MODEL_TO_FULLNAME,
)
from utils.logger import setup_logger
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen2_5_methods import get_response_qwen2_5, get_attention_qwen2_5
from qwen2_methods import get_response_qwen2, get_attention_qwen2
from chameleon_methods import get_attention_chameleon, get_response_chameleon
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from transformers.models.chameleon.processing_chameleon import ChameleonProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REFOCUS = "<REFOCUS>"


def calculate_entropy(att_map):
    flat_att = att_map.flatten()

    att_max = np.max(flat_att)
    exp_att = np.exp(flat_att - att_max)
    prob_dist = exp_att / (np.sum(exp_att) + 1e-10)
    entropy = max(-np.sum(prob_dist * np.log2(prob_dist + 1e-10)), 1e-12)
    return entropy


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def generation_mathvista_origin(model_name, data_dir, output_file_path):
    print(f"Saving results to {output_file_path}")
    model, processor = get_model(model_name)
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
        response = get_response(
            model_name=model_name,
            user_prompt=query,
            image_path=image_path,
            model=model,
            processor=processor,
        )
        results[problem_id] = problem
        results[problem_id]["query"] = query
        results[problem_id]["response"] = response

        save_every = 5
        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)

    print("MathVista: Generating Responses - Finish")


def generation_mathvista_refocus(model_name, output_file_path, input_file):
    print(f"Saving results to {output_file_path}")
    model, processor = get_model(model_name)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    print(f"Loading {input_file}...")
    results = read_json(input_file)
    full_pids = list(results.keys())
    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    skip_pids = []
    for pid, problem in existing_results.items():
        refocus_position = problem.get("refocus_position")
        refocus_response = problem.get("refocus_response")
        if refocus_position is not None and refocus_response is not None:
            results[pid]["refocus_position"] = refocus_position
            results[pid]["refocus_response"] = refocus_response
            skip_pids.append(problem["pid"])
    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid refocus_response. Skipping these problems..."
        )
    test_pids = [pid for pid in full_pids if pid not in skip_pids]
    print(f"Number of test problems to run: {len(test_pids)}")

    for i, pid in enumerate(tqdm(test_pids, desc="Generating refocus response")):
        problem = results[pid]
        query = problem["query"]
        image_path = os.path.join("data/mathvista", problem["image"])
        ori_response = problem["response"]

        refocus_position, refocus_response = refocus(
            model_name=model_name,
            model=model,
            processor=processor,
            user_prompt=query,
            image_path=image_path,
            ori_response=ori_response,
        )
        results[pid]["refocus_position"] = refocus_position
        results[pid]["refocus_response"] = refocus_response
        save_every = 1
        if (i % save_every == 0 and i > 0) or i == len(full_pids) - 1:
            save_json(results, output_file_path)


def generation_m3cot_origin(model_name, data_dir, output_file_path):
    print(f"Saving results to {output_file_path}")
    model, processor = get_model(model_name)
    data_file = os.path.join(data_dir, "test.jsonl")
    if not os.path.exists(data_file):
        raise ValueError(f"Data file {data_file} not found")
    data = open(data_file).readlines()
    data = [json.loads(d) for d in data]
    data = {d["id"]: d for d in data}
    full_ids = list(data.keys())
    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_ids = []
    for problem_id in full_ids:
        if problem_id in results and "response" in results[problem_id]:
            response = results[problem_id]["response"]
            if verify_response(response):
                skip_ids.append(problem_id)

    if len(skip_ids) > 0:
        print(
            f"Found existing results file with {len(skip_ids)} problems with valid responses. Skipping these problems..."
        )

    test_ids = [id for id in full_ids if id not in skip_ids]

    for i, problem_id in enumerate(tqdm(test_ids, desc="Generating origin response")):
        problem = data[problem_id]
        image_path = os.path.join(data_dir, problem["image"])
        query = (
            "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, after reasoning step by step based on the image."
            + "\nQuestion: {}"
            # + "\nChoices:"
            # + "\n(A) {}"
            # + "\n(B) {}"
            # + "\n(C) {}"
            # + "\n(D) {}"
        ).format(problem["question"])
        for ic, c in zip(["A", "B", "C", "D", "E", "F"], problem["choices"]):
            query += "\n({}) {}\n".format(ic, c)
        response = get_response(
            model_name=model_name,
            user_prompt=query,
            image_path=image_path,
            model=model,
            processor=processor,
        )
        results[problem_id] = problem
        results[problem_id]["query"] = query
        results[problem_id]["response"] = response
        save_every = 1
        if (i % save_every == 0 and i > 0) or i == len(test_ids) - 1:
            save_json(results, output_file_path)

    print("M3CoT: Generating Responses - Finish")


def generation_m3cot_refocus(model_name, output_file_path, input_file):
    print(f"Saving results to {output_file_path}")
    model, processor = get_model(model_name)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    print(f"Loading {input_file}...")
    results = read_json(input_file)
    full_pids = list(results.keys())
    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    skip_ids = []
    for id, problem in existing_results.items():
        refocus_position = problem.get("refocus_position")
        refocus_response = problem.get("refocus_response")
        if refocus_position is not None and refocus_response is not None:
            results[id]["refocus_position"] = refocus_position
            results[id]["refocus_response"] = refocus_response
            skip_ids.append(problem["id"])
    if len(skip_ids) > 0:
        print(
            f"Found existing results file with {len(skip_ids)} problems with valid refocus_response. Skipping these problems..."
        )
    test_ids = [id for id in full_pids if id not in skip_ids]
    print(f"Number of test problems to run: {len(test_ids)}")

    for i, id in enumerate(tqdm(test_ids, desc="Generating refocus response")):
        problem = results[id]
        query = problem["query"]
        image_path = os.path.join("data/m3cot", problem["image"])
        ori_response = problem["response"]
        refocus_position, refocus_response = refocus(
            model_name=model_name,
            model=model,
            processor=processor,
            user_prompt=query,
            image_path=image_path,
            ori_response=ori_response,
        )
        results[id]["refocus_position"] = refocus_position
        results[id]["refocus_response"] = refocus_response
        save_every = 1
        if (i % save_every == 0 and i > 0) or i == len(full_pids) - 1:
            save_json(results, output_file_path)


def get_model(model_name):
    if model_name == "qwen2_5":
        print(f"Loading {MODEL_TO_FULLNAME[model_name]} on {DEVICE}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_TO_FULLNAME[model_name],
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_TO_FULLNAME[model_name])
    elif model_name == "qwen2":
        print(f"Loading {MODEL_TO_FULLNAME[model_name]} on {DEVICE}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_TO_FULLNAME[model_name],
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_TO_FULLNAME[model_name])
    elif model_name == "chameleon":
        print(f"Loading {MODEL_TO_FULLNAME[model_name]} on {DEVICE}")
        model = ChameleonForConditionalGeneration.from_pretrained(
            MODEL_TO_FULLNAME[model_name],
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        processor = ChameleonProcessor.from_pretrained(MODEL_TO_FULLNAME[model_name])
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model, processor


def get_response(
    model_name, user_prompt, image_path, model, processor, cur_generation=None
):
    if model_name == "qwen2_5":
        return get_response_qwen2_5(
            user_prompt=user_prompt,
            image_path=image_path,
            model=model,
            processor=processor,
            cur_generation=cur_generation,
        )
    elif model_name == "qwen2":
        return get_response_qwen2(
            user_prompt=user_prompt,
            image_path=image_path,
            model=model,
            processor=processor,
            cur_generation=cur_generation,
        )
    elif model_name == "chameleon":
        return get_response_chameleon(
            user_prompt=user_prompt,
            image_path=image_path,
            model=model,
            processor=processor,
            cur_generation=cur_generation,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_attention(
    model_name,
    image_path,
    user_prompt,
    cur_generation,
    general_prompt,
    model,
    processor,
):
    if model_name == "qwen2_5":
        return get_attention_qwen2_5(
            image_path=image_path,
            user_prompt=user_prompt,
            cur_generation=cur_generation,
            general_prompt=general_prompt,
            model=model,
            processor=processor,
        )
    elif model_name == "qwen2":
        return get_attention_qwen2(
            image_path=image_path,
            user_prompt=user_prompt,
            cur_generation=cur_generation,
            general_prompt=general_prompt,
            model=model,
            processor=processor,
        )
    elif model_name == "chameleon":
        return get_attention_chameleon(
            image_path=image_path,
            user_prompt=user_prompt,
            cur_generation=cur_generation,
            general_prompt=general_prompt,
            model=model,
            processor=processor,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def refocus(model_name, model, processor, user_prompt, image_path, ori_response):
    model.eval()
    general_prompt = "Write a general description of the image."
    splited_response = split_sentence(ori_response)
    # 计算每个句子的注意力
    atts = []
    for sentence_idx in tqdm(
        range(len(splited_response)),
        desc="Calculating attention of each splited sentence.",
    ):
        if sentence_idx > 100:
            break
        cur_generation = "\n\n".join(splited_response[: sentence_idx + 1])
        att_map = get_attention(
            model_name=model_name,
            image_path=image_path,
            user_prompt=user_prompt,
            cur_generation=cur_generation,
            general_prompt=general_prompt,
            model=model,
            processor=processor,
        )  # shape: (num_path_width, num_path_height) (11, 23)
        # 计算均值和熵
        att_map_mean = np.mean(att_map)
        att_map_entropy = calculate_entropy(att_map)
        atts.append((sentence_idx + 1, att_map, att_map_mean, att_map_entropy))
    refocus_positions = []
    for i, (refocus_position, att_map, att_map_mean, att_map_entropy) in enumerate(
        atts
    ):
        print(f"Sentence {i}: {att_map_mean} {att_map_entropy}")
        if i == 0:
            att_map_mean_former, att_map_entropy_former = att_map_mean, att_map_entropy
            continue
        if (
            att_map_mean < att_map_mean_former
            and att_map_entropy > att_map_entropy_former
        ):
            # att_mean 减少, att_entropy 增加
            refocus_positions.append(refocus_position)
        att_map_mean_former, att_map_entropy_former = att_map_mean, att_map_entropy
    # 选取中间一个 refocus_position 重新生成 response
    print(f"Current available refocus positions: {refocus_positions}")
    if len(refocus_positions) == 0:
        # 如果没有合适的refocus_position，则直接将att_map_mean最低的地方设置为refocus_position
        refocus_position = atts[np.argmin([x[2] for x in atts])][0]
    else:
        refocus_position = refocus_positions[len(refocus_positions) // 2]
    refocus_response = splited_response[:refocus_position] + [REFOCUS]
    refocus_response = " ".join(refocus_response)
    refocus_response = get_response(
        model_name=model_name,
        user_prompt=user_prompt,
        image_path=image_path,
        model=model,
        processor=processor,
        cur_generation=refocus_response,
    )
    return refocus_position, refocus_response


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    period = args.period
    if period == "origin":
        output_dir = os.path.join("outputs_origin_old", dataset, model_name)
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
                "outputs_origin_old", dataset, model_name, "extracted_answer.json"
            )
            generation_mathvista_refocus(
                model_name=model_name,
                output_file_path=output_file_path,
                input_file=input_file,
            )
    elif dataset == "m3cot":
        output_file_path = os.path.join(output_dir, "generated_response.json")
        data_dir = os.path.join("data", "m3cot")
        if period == "origin":
            generation_m3cot_origin(
                model_name=model_name,
                data_dir=data_dir,
                output_file_path=output_file_path,
            )
        elif period == "refocus":
            input_file = os.path.join(
                "outputs_origin_old", dataset, model_name, "extracted_answer.json"
            )
            generation_m3cot_refocus(
                model_name=model_name,
                output_file_path=output_file_path,
                input_file=input_file,
            )
    else:
        raise ValueError(f"Dataset {dataset} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="qwen2_5", choices=MODEL_TO_FULLNAME.keys()
    )
    parser.add_argument(
        "--dataset", type=str, default="mathvista", choices=DATASET_TO_FULL_NAME.keys()
    )
    parser.add_argument(
        "--period",
        type=str,
        default="origin",
        choices=["origin", "refocus"],
    )
    args = parser.parse_args()
    main(args)
