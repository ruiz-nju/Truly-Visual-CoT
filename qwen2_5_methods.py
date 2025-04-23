from utils.tools import encode_base64
import torch
from qwen_vl_utils import process_vision_info
from utils.tools import split_sentence
import pdb
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFOCUS = "I should refocus on the image."


def prepare_qwen2_5_input(messages, processor):
    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def get_response_qwen2_5(
    user_prompt, image_path, model, processor, existing_generation=None
):
    image = Image.open(image_path)
    image_str = encode_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{image_str}"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    if existing_generation:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": existing_generation}],
            }
        )
    inputs = prepare_qwen2_5_input(messages=messages, processor=processor).to(
        DEVICE, torch.bfloat16
    )
    generate_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generation = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return generation.split("\nassistant\n")[-1].strip()


def refocus_qwen2_5(model, processor, user_prompt, image_path, ori_response):
    image = Image.open(image_path)
    image_str = encode_base64(image)
    model.eval()
    general_prompt = "Write a general description of the image."
    splited_response = split_sentence(ori_response)
    # 计算每个句子的注意力
    atts = []
    for sentence_idx in range(len(splited_response)):
        cur_generation = " ".join(splited_response[: sentence_idx + 1])
        att_map = get_attention_qwen2_5(
            image_str=image_str,
            prompt=user_prompt,
            cur_generation=cur_generation,
            general_prompt=general_prompt,
            model=model,
            processor=processor,
        )  # shape: (num_path_width, num_path_height) (11, 23)
        # 计算均值和熵
        att_map_mean = np.mean(att_map)
        att_map_entropy = calculate_entropy(att_map)
        atts.append((sentence_idx + 1, att_map, att_map_mean, att_map_entropy))
    refocus_positions = []
    for i, (refocus_position, att_map, att_map_mean, att_map_entropy) in enumerate(
        atts
    ):
        print(f"Sentence {i}: {att_map_mean} {att_map_entropy}")
        if i == 0:
            att_map_mean_former, att_map_entropy_former = att_map_mean, att_map_entropy
            continue
        if (
            att_map_mean < att_map_mean_former
            and att_map_entropy < att_map_entropy_former
        ):
            refocus_positions.append(refocus_position)
        att_map_mean_former, att_map_entropy_former = att_map_mean, att_map_entropy

    # 选取中间一个 refocus_position 重新生成 response
    refocus_position = refocus_positions[len(refocus_positions) // 2]
    refocus_response = splited_response[:refocus_position] + [REFOCUS]
    refocus_response = " ".join(refocus_response)
    refocus_response = get_response_qwen2_5(
        user_prompt=user_prompt,
        image_path=image_path,
        model=model,
        processor=processor,
        existing_generation=refocus_response,
    )
    return refocus_position, refocus_response


def calculate_entropy(att_map):
    flat_att = att_map.flatten()

    att_max = np.max(flat_att)
    exp_att = np.exp(flat_att - att_max)
    prob_dist = exp_att / (np.sum(exp_att) + 1e-10)
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

    return entropy


def get_attention_qwen2_5(
    image_str, prompt, cur_generation, general_prompt, model, processor
):

    # 计算cur_sentence attention
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": cur_generation},
            ],
        },
    ]
    general_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": general_prompt},
            ],
        }
    ]
    # 输入预处理
    inputs = prepare_qwen2_5_input(messages, processor).to(DEVICE, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(
        DEVICE, torch.bfloat16
    )

    # inputs["image_grid_thw"]: tensor([[ 1, 24, 38]], device='cuda:0')
    # 记录了每个图片的 patch 信息，包括 T, H, W，这里取 H, W
    # att_shape: 图像的 patch 数量 [12, 19]
    att_shape = (inputs["image_grid_thw"][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )  # 151652
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_end|>"
    )  # 151653

    # 定位视觉标记位置
    input_ids = inputs["input_ids"].tolist()[0]  # 获取所有 token
    pos = input_ids.index(vision_start_token_id) + 1  # 图像起始
    pos_end = input_ids.index(vision_end_token_id)  # 图像结束

    # 计算注意力
    ATT_LAYER = 22
    outputs = model(
        **inputs,
        output_attentions=True,  # 这里会输出所有层的attention，导致爆显存，看了Qwen的源码，没有找到支持输出指定层attention的方法，只能暂时在下面手动清空缓存并提取指定层
        output_hidden_states=False,
        use_cache=False,
    )
    att = (
        outputs["attentions"][ATT_LAYER][
            0, :, -1, pos:pos_end
        ]  # [heads, num_image_tokens]
        .mean(dim=(0))  # 平均多头
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    del outputs
    torch.cuda.empty_cache()
    general_outputs = model(
        **general_inputs,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
    )
    general_att = (
        general_outputs["attentions"][ATT_LAYER][
            0, :, -1, pos:pos_end
        ]  # [heads, num_image_tokens]
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
