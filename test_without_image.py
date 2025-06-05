import json
import os
from utils.tools import read_json, DEVICE, MAX_TOKENS, save_json, API_MODEL
from generate_response import get_model
import torch
import argparse
from openai import OpenAI
from extract_answer import (
    GPT_Model,
    extract_answer,
    normalize_extracted_answer,
    safe_equal,
)
from tqdm import tqdm

dataset = "mathvision"
model_name = "qwen2"
output_dir = f"outputs_without_image/{dataset}/{model_name}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def generate():
    print(f"Generating without image for {model_name} on {dataset}...")
    input_file = f"/mnt/hdd/zhurui/code/Truly-Visual-CoT/outputs_refocus/{dataset}/{model_name}/extracted_answer.json"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    output_file = os.path.join(output_dir, "generated_response.json")
    model, processor = get_model(model_name)

    data = read_json(input_file)

    for i, (id, problem) in enumerate(data.items()):
        print(f"********* Problem id: {id} [{i}/{len(data)}] *********")
        # refocus_response<REFOCUS>处截断
        cur_generation = problem.get("refocus_response").split("<REFOCUS>")[0]
        query = problem.get("query")
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            },
        ]
        text = processor.apply_chat_template(
            user_message, tokenize=False, add_generation_prompt=True
        )
        text = text + cur_generation
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE, torch.bfloat16)
        generate_ids = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, do_sample=False
        )
        generation = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = generation.split("\nassistant\n")[-1].strip()
        data[id]["without_image_response"] = response
        save_every = 1
        if (i % save_every == 0 and i > 0) or i == len(data) - 1:
            save_json(data, output_file)


def extract():
    input_file = os.path.join(output_dir, "generated_response.json")
    print(f"Extracting answers from {input_file}")
    output_file = os.path.join(output_dir, "extracted_answer.json")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.environ.get("ARK_API_KEY"),
    )
    extractor = GPT_Model(client=client, model=API_MODEL["doubao"])
    print(f"Extracting answers from {input_file}")
    data = read_json(input_file)
    if dataset == "mathvision":
        for i, (id, problem) in enumerate(data.items()):
            print(f"********* Problem id: {id} [{i}/{len(data)}] *********")
            response = problem.get("without_image_response")
            query = problem.get("query")
            choices = problem.get("options")
            question_type = "multi_choice" if len(choices) > 1 else None
            answer_type = None if len(choices) == 1 else "integer"
            precision = None
            extraction = extract_answer(
                question_type=question_type,
                answer_type=answer_type,
                choices=choices,
                query=query,
                pid=id,
                extractor=extractor,
                response=response,
            )
            data[id]["without_image_extraction"] = extraction
            prediction = normalize_extracted_answer(
                choices=choices,
                question_type=question_type,
                answer_type=answer_type,
                precision=precision,
                extraction=extraction,
            )
            answer = problem["answer"]
            true_false = safe_equal(extraction, answer)
            data[id]["without_image_prediction"] = prediction
            data[id]["without_image_true_false"] = true_false
            save_every = 1
            if (i % save_every == 0 and i > 0) or i == len(data) - 1:
                save_json(data, output_file)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def test():
    print(f"Testing without image for {model_name} on {dataset}...")
    input_file = os.path.join(output_dir, "extracted_answer.json")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    output_file = os.path.join(output_dir, "calculate_score.json")
    results = read_json(input_file)
    test_pids = list(results.keys())
    print(f"Number of test problems: {len(test_pids)}")
    print("Calculating the average accuracy...")
    total = len(test_pids)
    correct = 0
    for pid in tqdm(test_pids, desc="Reading results"):
        true_false = results[pid]["without_image_true_false"]
        if true_false:
            correct += 1

    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
    save_json(scores, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=3)
    args = parser.parse_args()
    period = args.period  # 1: generate, 2: extract, 3: test
    if period == 1:
        generate()
    elif period == 2:
        extract()
    elif period == 3:
        test()
