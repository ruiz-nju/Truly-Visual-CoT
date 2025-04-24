import json
import base64
from PIL import Image
from io import BytesIO
import re
import time

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
}

dataset_to_full_name = {"mathvista": "AI4Math/MathVista"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        data_json = json.dumps(data, indent=4)
        f.write(data_json)


def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def split_sentence(response):
    sentences = re.split(r"\n+", response)
    splited_response = [s.strip() for s in sentences if s.strip()]
    return splited_response


def print_info(module_name, model_name, dataset_name, period):
    print(
        f"""
    "================================================================"
    "🚀 Running  {module_name}"
    "📦 Model:   {model_to_fullname[model_name]}"
    "📚 Dataset: {dataset_to_full_name[dataset_name]}"
    "📅 Period:  {period}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )
