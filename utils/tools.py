import json
import base64
from PIL import Image
from io import BytesIO
import re


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
    sentence_delimiters = (
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.!?。！？])\s*"
    )
    sentences = re.split(sentence_delimiters, response)
    splited_response = [s.strip() for s in sentences if s.strip()]
    return splited_response
