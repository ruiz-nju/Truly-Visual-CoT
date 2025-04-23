import argparse
import os
from utils.tools import read_json
from utils.logger import setup_logger
import time
import torch
import time
from tqdm import tqdm
import pandas as pd
import json

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
}
dataset_to_full_name = {"mathvista": "AI4Math/MathVista"}


def print_info(model_name, dataset_name):
    print(
        f"""
    "================================================================"
    "🚀 Running calculate_score.py"
    "📦 Model:   {model_name}"
    "📚 Dataset: {dataset_name}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )


def get_full_metrics_str(metrics_dict) -> str:
    divider = "=" * 40

    avg_accuracy = metrics_dict["average"]["accuracy"]
    avg_correct = metrics_dict["average"]["correct"]
    avg_total = metrics_dict["average"]["total"]

    metrics_str = f"""
{f"Correct: {avg_correct}/{avg_total} - Accuracy: {avg_accuracy * 100:.2f}%"}
{divider}
""".lstrip()

    for key, item in metrics_dict.items():
        if key == "average":
            continue

        formatted_item_dict = {}
        for sub_key, sub_item in item.items():
            acc = sub_item["accuracy"]
            correct = sub_item["correct"]
            total = sub_item["total"]
            values = [f"{acc * 100:.2f}%", f"({correct}/{total})"]

            formatted_item_dict[sub_key] = values

        category_df = pd.DataFrame(
            formatted_item_dict, index=["Accuracy", "Correct/Total"]
        )

        metrics_str += f"""
{key}
{divider}
{category_df.T}
"""

    return metrics_str


def get_acc_with_contion(res_pd, key, value):
    if key == "skills":
        # if value in res_pd[key]:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd["true_false"] == True]
    acc = len(correct_pd) / len(total_pd)

    return len(correct_pd), len(total_pd), acc


def calculation_mathvista(
    input_file_path,
    output_file_path,
):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File {input_file_path} not found")
    print(f"Calculating score for {input_file_path}...")
    results = read_json(input_file_path)
    test_pids = list(results.keys())
    print(f"Number of test problems: {len(test_pids)}")
    print("Calculating the average accuracy...")
    total = len(test_pids)
    correct = 0
    for pid in tqdm(test_pids, desc="Reading results"):
        true_false = results[pid]["true_false"]
        if true_false:
            correct += 1

    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for pid in results:
        results[pid].update(results[pid].pop("metadata"))

    results_df = pd.DataFrame(results).T

    # asign the target keys for evaluation
    target_keys = [
        "question_type",
        "answer_type",
        "language",
        "source",
        "category",
        "task",
        "context",
        "grade",
        "skills",
    ]

    for key in target_keys:
        # get the unique values of the key
        if key == "skills":
            # the value is a list
            values = []
            for i in range(len(results_df)):
                values += results_df[key][i]
            values = list(set(values))
        else:
            values = results_df[key].unique()

        # calculate the accuracy for each value
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(results_df, key, value)
            if total > 0:
                scores[key][value] = {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }

        # sort the scores by accuracy
        scores[key] = dict(
            sorted(
                scores[key].items(),
                key=lambda item: float(item[1]["accuracy"]),
                reverse=True,
            )
        )

    metrics_str = get_full_metrics_str(scores)
    print(metrics_str)

    print(f"Saving scores to {output_file_path}")
    with open(output_file_path, "w") as f:
        json.dump(scores, f, indent=4)

    print("MathVista: Calculating Scores - Finish")


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    output_dir = os.path.join("outputs", dataset, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    setup_logger(os.path.join(output_dir, "logs", "calculate_score_log.txt"))
    print_info(model_name, dataset)
    if dataset == "mathvista":
        output_file_path = os.path.join(output_dir, "calculated_score.json")
        input_file_path = os.path.join(output_dir, "extracted_answer.json")
        calculation_mathvista(
            input_file_path=input_file_path,
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
    args = parser.parse_args()
    main(args)
