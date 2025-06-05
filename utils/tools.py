import json
import base64
from PIL import Image
from io import BytesIO
import re
import time
import torch
import pdb

MODEL_TO_FULLNAME = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
    "chameleon": "facebook/chameleon-7b",
}

DATASET_TO_FULL_NAME = {
    "mathvista": "AI4Math/MathVista",
    "m3cot": "LightChen2333/M3CoT",
    "mathvision": "MathLLMs/MathVision",
}

API_MODEL = {
    "doubao": "ep-20250512113505-sd5t2",
    "deepseek": "ep-20250422234405-ddr6w",
}

USE_EAMPLE = True

MAX_IMAGE_SIZE = 1024

EXAMPLE_IMAGE_PATH = (
    "/mnt/hdd/zhurui/code/Truly-Visual-CoT/data/mathvista/images/980.jpg"
)
EAMPLE_PROMPT = "When you output <REFOCUS>, revisit the picture and check some key conditions that are relevant to the reasoning at hand. you should Here is an example question and answer pair that can be used for reference.\n\nExample question:\n\nWhat is the highest number shown?\n\nExample response:\n\nTo determine the highest number shown on the clock in the image:\n\n1. Identify the numbers on the clock face.\n2. Compare these numbers to find the largest one. <REFOCUS> The clock face shows the numbers 1 through 12, which are standard for clock faces. The highest number among these is 12.\n\nFinal answer: 12"

MAX_TOKENS = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(img_path, resize=False):
    img = Image.open(img_path)
    width, height = img.size

    if resize and max(width, height) > MAX_IMAGE_SIZE:
        scaling_factor = MAX_IMAGE_SIZE / float(max(width, height))
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        print(
            f"Image {img_path} is resized to {new_width}x{new_height} from {width}x{height}"
        )

    return img


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
    # 匹配行内公式（$...$）和显示公式（$$...$$）
    formula_pattern = r"(\\$.*?\\$|\\$$.*?\\$$)"
    formulas = []

    def replace_formula(match):
        formulas.append(match.group(0))
        return f"__FORMULA_{len(formulas)-1}__"

    # 用占位符替换所有公式，避免干扰分句
    text_with_placeholders = re.sub(
        formula_pattern, replace_formula, response, flags=re.DOTALL
    )

    # 分割处理：先按换行符分块，再按句子结束符分句
    sentences = []
    blocks = [b.strip() for b in text_with_placeholders.split("\n") if b.strip()]

    for block in blocks:
        # 在每块中按句子结束符（.!?）分割，同时保留结尾符号
        split_points = re.finditer(r"([.!?])\s+", block)
        last_idx = 0
        for match in split_points:
            end = match.end()
            sentences.append(
                block[last_idx : end - 1].strip()
            )  # -1保留标点，去除末尾空格
            last_idx = end
        # 添加剩余部分
        if last_idx < len(block):
            sentences.append(block[last_idx:].strip())

    # 恢复公式内容
    for i in range(len(formulas)):
        placeholder = f"__FORMULA_{i}__"
        for j in range(len(sentences)):
            sentences[j] = sentences[j].replace(placeholder, formulas[i])

    return sentences


def print_info(module_name, model_name, dataset_name, period):
    print(
        f"""
    "================================================================"
    "🚀 Running  {module_name}"
    "📦 Model:   {MODEL_TO_FULLNAME[model_name]}"
    "📚 Dataset: {DATASET_TO_FULL_NAME[dataset_name]}"
    "📅 Period:  {period}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )
