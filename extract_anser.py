import argparse
import logging
import os
import re
import pdb
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image
from utils.tools import (
    read_json,
    save_json,
    print_info,
    DATASET_TO_FULL_NAME,
    MODEL_TO_FULLNAME,
)
from utils.logger import setup_logger
import time
import torch
import base64
import time
from io import BytesIO
from typing import Union
from tqdm import tqdm
from Levenshtein import distance

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""
EXTRACTOR = "ep-20250422234405-ddr6w"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# build gpt class
class GPT_Model:
    def __init__(
        self,
        client: Union[OpenAI, AzureOpenAI],
        model="ep-20250422234405-ddr6w",  # 这里用的是DeepSeek V3
        temperature=0,
        max_tokens=1024,
        n=1,
        patience=1000000,
        sleep_time=0,
    ):
        self.client = client
        self.model = model
        self.use_image = True if "vision" in model else False
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time

    def get_response(
        self, user_prompt: str, decoded_image: Union[Image.Image, None] = None
    ):
        patience = self.patience
        max_tokens = self.max_tokens

        user_messages: list[ChatCompletionContentPartParam] = []

        if self.use_image:
            if decoded_image is None:
                print(
                    f"You are using a model that supports vision: {self.model}, "
                    f"but no image was provided when generating a response. This is likely unintended."
                )
            else:
                buffered = BytesIO()
                decoded_image.save(buffered, format="PNG")
                base64_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                user_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_str}"
                        },
                    }
                )

        user_messages.append({"type": "text", "text": user_prompt})

        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": user_messages},
        ]

        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=self.n,
                )

                predictions = [choice.message.content for choice in response.choices]
                prediction = predictions[0]

                if prediction != "" and prediction is not None:
                    return prediction.strip()

            except Exception as e:
                print(e)

                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!!Reduce max_tokens to", max_tokens)
                if max_tokens < 8:
                    return ""
                if "Please reduce the length of the messages." in str(e):
                    print("!!Reduce user_prompt to", user_prompt[:-1])
                    return ""
                if self.sleep_time > 0:
                    print(f"Sleeping for {self.sleep_time} seconds")
                    time.sleep(self.sleep_time)

        return ""


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(
    question_type,
    answer_type,
    choices,
    query,
    pid,
    extractor,
    response,
    quick_extract=False,
):

    if response == "":
        return ""

    if question_type == "multi_choice" and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(DEMO_PROMPT, query, response)
        extraction = extractor.get_response(user_prompt=full_prompt)
        return extraction
    except Exception as e:
        logging.info(
            f"Error in extracting answer for problem: {pid} with response: {response}"
        )
        logging.info(e)

    return ""


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        logging.info(e)
        return False


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(
    choices, question_type, answer_type, precision, extraction
):
    if question_type == "multi_choice":
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord("A") + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == "integer":
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "float":
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "list":
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def extraction_mathvista_origin(
    input_file_path, output_file_path, extractor_name="ep-20250422234405-ddr6w"
):
    target = "response"

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File {input_file_path} not found")
    print(f"Saving results to {output_file_path}")
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",  # 这里使用的是 DeepSeek V3 的api
        api_key=os.environ.get("ARK_API_KEY"),
    )

    extractor = GPT_Model(client=client, model=extractor_name)
    print(f"Extracting answers from {input_file_path}")

    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    print(f"Loading {input_file_path}...")
    results = read_json(input_file_path)
    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in existing_results.items():
        prediction = problem.get("prediction")
        true_false = problem.get("true_false")
        if prediction is not None and true_false is not None:
            results[pid]["prediction"] = prediction
            results[pid]["true_false"] = true_false
            skip_pids.append(problem["pid"])

    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid prediction and true_false. Skipping these problems..."
        )
    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    print(f"Number of test problems to run: {len(test_pids)}")

    save_every = 5
    for i, pid in enumerate(tqdm(test_pids, desc="Extracting answers")):
        problem = results[pid]
        assert target in problem, f"Label '{target}' not found in problem"
        response = problem[target]
        extraction = extract_answer(
            question_type=problem["question_type"],
            answer_type=problem["answer_type"],
            choices=problem["choices"],
            query=problem["query"],
            pid=pid,
            extractor=extractor,
            response=response,
        )
        # 将提取的答案保存到results中
        results[pid]["extraction"] = extraction
        # 将提取的答案标准化
        choices = problem["choices"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        precision = problem["precision"]
        prediction = normalize_extracted_answer(
            choices=choices,
            question_type=question_type,
            answer_type=answer_type,
            precision=precision,
            extraction=extraction,
        )
        answer = problem["answer"]
        true_false = safe_equal(prediction, answer)
        results[pid]["prediction"] = prediction
        results[pid]["true_false"] = true_false

        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)

    print("MathVista: Extract Answers - Finish")


def extraction_mathvista_refocus(input_file_path, output_file_path):
    target = "refocus_response"

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File {input_file_path} not found")
    print(f"Saving results to {output_file_path}")
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",  # 这里使用的是 DeepSeek V3 的api
        api_key=os.environ.get("ARK_API_KEY"),
    )

    extractor = GPT_Model(client=client, model=EXTRACTOR)
    print(f"Extracting answers from {input_file_path}")

    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    print(f"Loading {input_file_path}...")
    results = read_json(input_file_path)
    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in existing_results.items():
        refocus_prediction = problem.get("refocus_prediction")
        refocus_true_false = problem.get("refocus_true_false")
        if refocus_prediction is not None and refocus_true_false is not None:
            results[pid]["refocus_prediction"] = refocus_prediction
            results[pid]["refocus_true_false"] = refocus_true_false
            skip_pids.append(problem["pid"])

    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid refocus_prediction and refocus_true_false. Skipping these problems..."
        )
    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    print(f"Number of test problems to run: {len(test_pids)}")

    save_every = 1
    for i, pid in enumerate(tqdm(test_pids, desc="Extracting answers")):
        problem = results[pid]
        assert target in problem, f"Label '{target}' not found in problem"
        refocus_response = problem[target]
        refocus_extraction = extract_answer(
            question_type=problem["question_type"],
            answer_type=problem["answer_type"],
            choices=problem["choices"],
            query=problem["query"],
            pid=pid,
            extractor=extractor,
            response=refocus_response,
        )
        # 将提取的答案保存到results中
        results[pid]["refocus_extraction"] = refocus_extraction
        # 将提取的答案标准化
        choices = problem["choices"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        precision = problem["precision"]
        refocus_prediction = normalize_extracted_answer(
            choices=choices,
            question_type=question_type,
            answer_type=answer_type,
            precision=precision,
            extraction=refocus_extraction,
        )
        answer = problem["answer"]
        refocus_true_false = safe_equal(prediction=refocus_prediction, answer=answer)
        results[pid]["refocus_prediction"] = refocus_prediction
        results[pid]["refocus_true_false"] = refocus_true_false

        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)

    print("MathVista: Extract Answers - Finish")


def extraction_m3cot_origin(input_file_path, output_file_path):
    target = "response"

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File {input_file_path} not found")
    print(f"Saving results to {output_file_path}")
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",  # 这里使用的是 DeepSeek V3 的api
        api_key=os.environ.get("ARK_API_KEY"),
    )

    extractor = GPT_Model(client=client, model=EXTRACTOR)
    print(f"Extracting answers from {input_file_path}")

    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    print(f"Loading {input_file_path}...")
    results = read_json(input_file_path)
    full_pids = list(results.keys())

    skip_pids = []
    for id, problem in existing_results.items():
        prediction = problem.get("prediction")
        true_false = problem.get("true_false")
        if prediction is not None and true_false is not None:
            results[id]["prediction"] = prediction
            results[id]["true_false"] = true_false
            skip_pids.append(problem["id"])

    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid prediction and true_false. Skipping these problems..."
        )
    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    print(f"Number of test problems to run: {len(test_pids)}")

    save_every = 5
    for i, id in enumerate(tqdm(test_pids, desc="Extracting answers")):
        problem = results[id]
        assert target in problem, f"Label '{target}' not found in problem"
        response = problem[target]
        query = problem["query"]
        choices = problem["choices"]
        question_type = "multi_choice"
        answer_type = None
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
        # 将提取的答案保存到results中
        results[id]["extraction"] = extraction
        # 将提取的答案标准化
        prediction = normalize_extracted_answer(
            choices=choices,
            question_type=question_type,
            answer_type=answer_type,
            precision=precision,
            extraction=extraction,
        )
        answer = problem["answer"]
        true_false = safe_equal(extraction, answer)
        results[id]["prediction"] = prediction
        results[id]["true_false"] = true_false

        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)

    print("M3CoT: Extract Answers - Finish")


def extraction_m3cot_refocus(input_file_path, output_file_path):
    target = "refocus_response"

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File {input_file_path} not found")
    print(f"Saving results to {output_file_path}")
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",  # 这里使用的是 DeepSeek V3 的api
        api_key=os.environ.get("ARK_API_KEY"),
    )

    extractor = GPT_Model(client=client, model=EXTRACTOR)
    print(f"Extracting answers from {input_file_path}")

    if os.path.exists(output_file_path):
        print(f"Loading existing {output_file_path}...")
        existing_results = read_json(output_file_path)
    else:
        existing_results = {}

    print(f"Loading {input_file_path}...")
    results = read_json(input_file_path)
    full_pids = list(results.keys())

    skip_pids = []
    for id, problem in existing_results.items():
        prediction = problem.get("refocus_prediction")
        true_false = problem.get("refocus_true_false")
        if prediction is not None and true_false is not None:
            results[id]["refocus_prediction"] = prediction
            results[id]["refocus_true_false"] = true_false
            skip_pids.append(problem["id"])

    if len(skip_pids) > 0:
        print(
            f"Found existing results file with {len(skip_pids)} problems with valid prediction and true_false. Skipping these problems..."
        )
    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    print(f"Number of test problems to run: {len(test_pids)}")

    save_every = 1
    for i, id in enumerate(tqdm(test_pids, desc="Extracting answers")):
        problem = results[id]
        assert target in problem, f"Label '{target}' not found in problem"
        refocus_response = problem[target]
        query = problem["query"]
        choices = problem["choices"]
        question_type = "multi_choice"
        answer_type = None
        precision = None
        # extraction = extract_answer(extractor, response, problem)
        extraction = extract_answer(
            question_type=question_type,
            answer_type=answer_type,
            choices=choices,
            query=query,
            pid=id,
            extractor=extractor,
            response=refocus_response,
        )
        # 将提取的答案保存到results中
        results[id]["refocus_extraction"] = extraction
        # 将提取的答案标准化
        prediction = normalize_extracted_answer(
            choices=choices,
            question_type=question_type,
            answer_type=answer_type,
            precision=precision,
            extraction=extraction,
        )
        answer = problem["answer"]
        true_false = safe_equal(extraction, answer)
        results[id]["refocus_prediction"] = prediction
        results[id]["refocus_true_false"] = true_false

        if (i % save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, output_file_path)

    print("M3CoT: Extract Answers - Finish")


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
    setup_logger(os.path.join(output_dir, "logs", "extract_answer_log.txt"))
    print_info(
        module_name="extract_answer.py",
        model_name=model_name,
        dataset_name=dataset,
        period=period,
    )
    if dataset == "mathvista":
        output_file_path = os.path.join(output_dir, "extracted_answer.json")
        input_file_path = os.path.join(output_dir, "generated_response.json")
        if period == "origin":
            extraction_mathvista_origin(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
            )
        elif period == "refocus":
            extraction_mathvista_refocus(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
            )
    elif dataset == "m3cot":
        output_file_path = os.path.join(output_dir, "extracted_answer.json")
        input_file_path = os.path.join(output_dir, "generated_response.json")
        if period == "origin":
            extraction_m3cot_origin(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
            )
        elif period == "refocus":
            extraction_m3cot_refocus(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
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
