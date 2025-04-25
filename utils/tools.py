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
    "📦 Model:   {model_to_fullname[model_name]}"
    "📚 Dataset: {dataset_to_full_name[dataset_name]}"
    "📅 Period:  {period}"
    "⏰ Time:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    "================================================================"
    """
    )
