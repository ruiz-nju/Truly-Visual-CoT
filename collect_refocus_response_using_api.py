import os
import pdb
from utils.tools import read_json, save_json, API_MODEL
from generate_response import get_model
from extract_answer import GPT_Model
from openai import OpenAI
import torch

DEMO_PROMPT = """
I will provide a question related to visual question answering (VQA), the correct answer, and the original response. Your task is to determine where in the original response the model should refocus on the image to prevent visual memory decay, and insert <REFOCUS> at those positions. You may add <REFOCUS> in multiple places if needed, but do not modify any other part of the original response. Your output should only contain the final modified response—no additional explanations or information.

Example Format:

Query:
Please solve the problem step by step based on the image and put your answer in one \"\\boxed{{}}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{{}}\".\nHow many different digits can you find in this picture?\n<image1>\n

Answer:
6

Original response:
To determine how many different digits can be found in the picture, let's follow these steps:\n\n1. Identify all the digits present in the image.\n2. Count the unique digits.\n\nStep 1: Identifying the digits:\n- The number \"5\" is visible in the image.\n\nStep 2: Counting the unique digits:\n- The digit \"5\" is the only digit present in the image.\n\nTherefore, there is only one unique digit in the image.\n\n\\boxed{{1}}

Model response:
To determine how many different digits can be found in the picture, let's follow these steps:\n\n1. Identify all the digits present in the image.\n2. Count the unique digits.\n\nStep 1: Identifying the digits:<REFOCUS>\n- The number \"5\" is visible in the image.\n\nStep 2: <REFOCUS>Counting the unique digits:\n- The digit \"5\" is the only digit present in the image.\n\nTherefore, there is only one unique digit in the image.\n\n\\boxed{{1}}


Now, please modify the response based on the given question, answer, and current response.


Query:
{query}

Answer:
{answer}

Original response:
{original_response}
"""

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)
extractor = GPT_Model(client=client, model=API_MODEL["doubao"])

model_name = "qwen2"
dataset = "mathvision"

input_file = f"outputs_origin_old/{dataset}/{model_name}/generated_response.json"
output_dir = f"outputs_collect_data/{dataset}/{model_name}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, "collected_response.json")

data = read_json(input_file)

for i, (id, item) in enumerate(data.items()):
    print(f"Processing problem {id} [{i+1}/{len(data)}]")
    query = item["query"]
    answer = item["answer"]
    original_response = item["response"]

    prompt = DEMO_PROMPT.format(
        query=query, answer=answer, original_response=original_response
    )
    response = extractor.get_response(user_prompt=prompt)
    data[id]["collected_response"] = response
    save_every = 1

    if (i % save_every == 0 and i > 0) or i == len(data) - 1:
        save_json(data, output_file)
