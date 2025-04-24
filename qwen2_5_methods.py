from utils.tools import encode_base64
import torch
from qwen_vl_utils import process_vision_info
from utils.tools import split_sentence
import pdb
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFOCUS = "Wait, I should refocus on the image to double-check some necessary details and continue my reasoning."


def prepare_qwen2_5_input(user_prompt, image_path, processor, cur_generation=None):
    """
    Prepare the input for Qwen2.5VL.
    """
    image = Image.open(image_path)
    image_str = encode_base64(image=image)
    user_message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    image_inputs, video_inputs = process_vision_info(user_message)
    text = processor.apply_chat_template(
        user_message, tokenize=False, add_generation_prompt=True
    )
    if cur_generation:
        text = text + cur_generation
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def get_response_qwen2_5(
    user_prompt, image_path, model, processor, cur_generation=None
):
    inputs = prepare_qwen2_5_input(
        user_prompt=user_prompt,
        image_path=image_path,
        processor=processor,
        cur_generation=cur_generation,
    ).to(DEVICE, torch.bfloat16)
    generate_ids = model.generate(**inputs, max_new_tokens=5096, do_sample=False)
    generation = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return generation.split("\nassistant\n")[-1].strip()


def refocus_qwen2_5(model, processor, user_prompt, image_path, ori_response):
    model.eval()
    general_prompt = "Write a general description of the image."
    splited_response = split_sentence(ori_response)
    # 计算每个句子的注意力
    atts = []
    for sentence_idx in range(len(splited_response)):
        cur_generation = " ".join(splited_response[: sentence_idx + 1])
        att_map = get_attention_qwen2_5(
            image_path=image_path,
            user_prompt=user_prompt,
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
            and att_map_entropy > att_map_entropy_former
        ):
            refocus_positions.append(refocus_position)
        att_map_mean_former, att_map_entropy_former = att_map_mean, att_map_entropy
    # 选取中间一个 refocus_position 重新生成 response
    if len(refocus_positions) == 0:
        # 如果没有合适的refocus_position，则直接将att_map_mean最低的地方设置为refocus_position
        refocus_position = atts[np.argmin([x[2] for x in atts])][0]
    else:
        refocus_position = refocus_positions[len(refocus_positions) // 2]
    refocus_response = splited_response[:refocus_position] + [REFOCUS]
    refocus_response = " ".join(refocus_response)
    refocus_response = get_response_qwen2_5(
        user_prompt=user_prompt,
        image_path=image_path,
        model=model,
        processor=processor,
        cur_generation=refocus_response,
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
    user_prompt, image_path, cur_generation, general_prompt, model, processor
):
    # 输入预处理
    inputs = prepare_qwen2_5_input(
        user_prompt=user_prompt,
        image_path=image_path,
        processor=processor,
        cur_generation=cur_generation,
    ).to(DEVICE, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(
        user_prompt=general_prompt,
        image_path=image_path,
        processor=processor,
    ).to(DEVICE, torch.bfloat16)

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
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            only_output_attention_one_layer=ATT_LAYER,  # 使用了这个参数的话，outputs输出的只有指定层的attention
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
