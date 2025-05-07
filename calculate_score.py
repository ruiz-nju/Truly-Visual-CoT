import argparse
import os
from utils.tools import read_json, print_info, DATASET_TO_FULL_NAME, MODEL_TO_FULLNAME
from utils.logger import setup_logger
import time
import torch
import time
from tqdm import tqdm
import pandas as pd
import json


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


def calculation(
    dataset_name,
    input_file_path,
    output_file_path,
    period,
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
        if period == "origin":
            true_false = results[pid]["true_false"]
        elif period == "refocus":
            true_false = results[pid]["refocus_true_false"]
        if true_false:
            correct += 1

    accuracy = correct / total
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
    if dataset_name == "mathvista":
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

    print("Calculating Scores - Finish")


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
    setup_logger(os.path.join(output_dir, "logs", "calculate_score_log.txt"))
    print_info(
        module_name=f"calculate_score.py",
        model_name=model_name,
        dataset_name=dataset,
        period=period,
    )
    output_file_path = os.path.join(output_dir, "calculated_score.json")
    input_file_path = os.path.join(output_dir, "extracted_answer.json")
    calculation(
        dataset_name=dataset,
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        period=period,
    )


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
