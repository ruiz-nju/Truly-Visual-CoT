from utils.tools import (
    encode_base64,
    load_image,
    USE_EAMPLE,
    EXAMPLE_IMAGE_PATH,
    EAMPLE_PROMPT,
    DEVICE,
    MAX_TOKENS,
)
import pdb
import torch
from qwen_vl_utils import process_vision_info
import numpy as np


def prepare_qwen2_input(
    user_prompt,
    image_path,
    processor,
    cur_generation=None,
    additional_image_path=None,
):
    image = load_image(image_path, resize=True)
    image_str = encode_base64(image=image)
    if USE_EAMPLE:
        example_image = load_image(EXAMPLE_IMAGE_PATH, resize=True)
        example_image_str = encode_base64(image=example_image)
        user_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{example_image_str}",
                    },
                    {"type": "text", "text": EAMPLE_PROMPT},
                    {"type": "image", "image": f"data:image;base64,{image_str}"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    else:
        user_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{image_str}",
                        "text": user_prompt,
                    },
                ],
            },
        ]
    text = processor.apply_chat_template(
        user_message, tokenize=False, add_generation_prompt=True
    )
    if cur_generation:
        text = text + cur_generation
    image_inputs, video_inputs = process_vision_info(user_message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def get_response_qwen2(
    user_prompt,
    image_path,
    model,
    processor,
    cur_generation,
):
    inputs = prepare_qwen2_input(
        user_prompt,
        image_path,
        processor,
        cur_generation,
    ).to(DEVICE, torch.bfloat16)
    generate_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    generation = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return generation.split("\nassistant\n")[-1].strip()


def get_attention_qwen2(
    image_path, user_prompt, cur_generation, general_prompt, model, processor
):
    # 输入预处理
    inputs = prepare_qwen2_input(
        user_prompt=user_prompt,
        image_path=image_path,
        processor=processor,
        cur_generation=cur_generation,
    ).to(DEVICE, torch.bfloat16)
    general_inputs = prepare_qwen2_input(
        user_prompt=general_prompt,
        image_path=image_path,
        processor=processor,
    ).to(DEVICE, torch.bfloat16)

    # inputs["image_grid_thw"]: tensor([[ 1, 24, 38]], device='cuda:0')
    # 记录了每个图片的 patch 信息，包括 T, H, W，这里取 H, W
    # att_shape: 图像的 patch 数量 [12, 19]
    if USE_EAMPLE:
        att_shape = (
            (inputs["image_grid_thw"][1, 1:] / 2).cpu().numpy().astype(int).tolist()
        )
    else:
        att_shape = (
            (inputs["image_grid_thw"][0, 1:] / 2).cpu().numpy().astype(int).tolist()
        )
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # 定位视觉标记位置
    input_ids = inputs["input_ids"].tolist()[0]  # 获取所有 token
    if USE_EAMPLE:
        # 获取所有vision_start_token_id的位置
        start_indices = [
            i for i, x in enumerate(input_ids) if x == vision_start_token_id
        ]
        # 获取所有vision_end_token_id的位置
        end_indices = [i for i, x in enumerate(input_ids) if x == vision_end_token_id]
        # 检查是否有足够的标记
        if len(start_indices) >= 2 and len(end_indices) >= 2:
            pos = start_indices[1] + 1  # 第二个图像起始位置（+1排除标记本身）
            pos_end = end_indices[1]  # 第二个图像结束位置
        else:
            raise ValueError("没有找到足够的视觉标记对")
    else:
        pos = input_ids.index(vision_start_token_id) + 1  # 图像起始
        pos_end = input_ids.index(vision_end_token_id)  # 图像结束

    # 计算注意力
    ATT_LAYER = 22
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            only_output_attention_one_layer=ATT_LAYER,
            output_hidden_states=False,
            use_cache=False,
        )  # outputs.shape: torch.Size([1, 28, 1714, 1714])
    att = (
        outputs[0, :, -1, pos:pos_end]  # [heads, num_image_tokens]
        .mean(dim=(0))  # 平均多头
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    del outputs
    torch.cuda.empty_cache()
    with torch.no_grad():
        general_outputs = model(
            **general_inputs,
            output_attentions=True,
            only_output_attention_one_layer=ATT_LAYER,
            output_hidden_states=False,
            use_cache=False,
        )
    general_att = (
        general_outputs[0, :, -1, pos:pos_end]  # [heads, num_image_tokens]
        .mean(dim=(0))  # 平均多头
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    del general_outputs
    torch.cuda.empty_cache()
    att_map = att / general_att
    att_map = att_map.reshape(att_shape)
    return att_map
