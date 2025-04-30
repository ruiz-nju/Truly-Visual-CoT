import os
import argparse
from ruamel.yaml import YAML

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default="./config/config.yaml", help="global environment configs"
)
args = parser.parse_args()
yaml = YAML()

# Reading a YAML file
with open(args.config, "r") as file:
    config = yaml.load(file)
    print(config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import argparse
from PIL import Image
from ruamel.yaml import YAML

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
    ModelOutput,
)

# default: Load the model on the available device(s)

import json
import copy
import random
from tqdm import tqdm

IMG_FOLDER = None  # TODO
EVAL_FILE = None  # TODO
DATA_NAME = "m3cot"


def calculate_entropy(att_map):
    flat_att = att_map.flatten()

    att_max = np.max(flat_att)
    exp_att = np.exp(flat_att - att_max)
    prob_dist = exp_att / (np.sum(exp_att) + 1e-10)
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

    return entropy


@dataclass
class Qwen2_5_VLCausalLMOutputWithPastForInterCoT(Qwen2_5_VLCausalLMOutputWithPast):
    selected_vokens: torch.LongTensor = None
    past_attentions: List = None


class Qwen2_5_VLForInterCoT(Qwen2_5_VLForConditionalGeneration):
    # def __init__(self, config):
    #     super().__init__(config)
    #     self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
    #     self.model = Qwen2_5_VLModel(config)
    #     self.vocab_size = config.vocab_size
    #     self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    #     self.rope_deltas = None  # cache rope_deltas here
    #
    #     # Initialize weights and apply final processing
    #     self.post_init()
    # 重写forward方法,使其能够计算熵
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        img_embeds_add: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPastForInterCoT]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                # import pdb; pdb.set_trace()
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            # added
            if img_embeds_add is not None:
                image_embeds = img_embeds_add
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # added inilize for kv cot
        if (
            output_attentions
            and (past_key_values.key_cache == [] or past_key_values.key_cache[0] == [])
            and pixel_values is not None
        ):
            self.new_tokens = 0
            self.num_selected_patches = num_selected_patches
            img_start_token_id = 151652  # 8197
            img_end_token_id = 151653
            self.query_image_start = (input_ids == img_start_token_id).nonzero(
                as_tuple=True
            )[1][-1] + 1
            self.query_image_end = (input_ids == img_end_token_id).nonzero(
                as_tuple=True
            )[1][-1]
            # self.boi_token, self.eoi_token = input_ids[:, self.query_image_start-1].unsqueeze(0), input_ids[:, self.query_image_end+1024].unsqueeze(0)
            with torch.no_grad():
                self.query_vokens = image_embeds

            self.query_image_mask = torch.zeros_like(
                input_ids, device=input_ids.device
            ).bool()
            self.query_image_mask[:, self.query_image_start : self.query_image_end] = (
                True  # self.query_image_mask[:, self.query_image_start: self.query_image_start+1024] = True
            )

            self.num_line_break = 0

        elif output_attentions:
            self.new_tokens += 1
            false_tensor = torch.tensor([[False]], device=self.query_image_mask.device)
            self.query_image_mask = torch.cat(
                [self.query_image_mask, false_tensor], dim=-1
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            # sub_image_masks=sub_image_masks,
            # pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Disallow image tokens which does not include special begin-image and end-image tokens
        # image_tokens = self.model.vocabulary_mapping.image_tokens

        selected_vokens = None
        dot_id = 13
        # 断句
        if output_attentions and input_ids[:, -1] == dot_id:
            self.num_line_break += 1

        if (
            pixel_values is None
            and output_attentions
            and self.num_line_break % 2 == 0
            and input_ids[:, -1] == dot_id
        ):
            attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]
            image_attentions = attentions[self.query_image_mask]

            att_map_mean = image_attentions.mean()
            att_map_entropy = calculate_entropy(image_attentions)
            self.past_attentions.append(
                (self.num_line_break, image_attentions, att_map_mean, att_map_entropy)
            )

            # 条件判断
            condition_met = True  # 初始默认允许插入
            if "past_attentions" not in self.__dict__:
                self.past_attentions = []
            if self.previous_area is not None and self.previous_attn_mean is not None:
                # 条件1：当前注意力 < 上一句注意力
                area_condition = att_map_mean < self.past_attentions[-1][2]
                # 条件2：当前注意力熵 > 上一句熵
                attn_condition = att_map_entropy > self.past_attentions[-1][3]
                condition_met = area_condition & attn_condition

            if condition_met:
                selected_vokens = torch.cat(
                    [  # self.boi_token,
                        self.query_vokens,
                        # self.eoi_token
                    ],
                    dim=-1,
                )
                # 注入新的token
                self.query_image_mask = torch.cat(
                    [
                        self.query_image_mask,
                        torch.zeros(
                            self.query_image_mask.shape[0],
                            self.query_vokens.shape[0] + 2,  #
                            device=self.query_image_mask.device,
                        ).bool(),
                    ],
                    dim=1,
                )
                # 更新历史记录
                self.past_attentions.append(
                    (
                        self.num_line_break,
                        image_attentions,
                        att_map_mean,
                        att_map_entropy,
                    )
                )

            else:
                # 不插入图像 Token，保留原有输入
                selected_vokens = None

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPastForInterCoT(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            selected_vokens=selected_vokens,
        )

    # 重写输入生成函数，需要将当前输入移动到新插入的图像token之前
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        img_embeds_add = None
        if "img_embeds_add" in kwargs:
            img_embeds_add = kwargs["img_embeds_add"].clone()
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                if (
                    "selected_vokens" in kwargs
                    and kwargs["selected_vokens"] is not None
                ):
                    if kwargs["selected_vokens"].dtype == torch.bfloat16:
                        # add_tokens = torch.ones_like(kwargs['selected_vokens'][0].unsqueeze(0),
                        #                              dtype=torch.int32) * 151655
                        # add_tokens = torch.cat([torch.tensor([[151652]], device=add_tokens.device), add_tokens,
                        #                         torch.tensor([[151653]], device=add_tokens.device)], dim=-1)
                        # input_ids = torch.cat([input_ids, add_tokens], dim=-1)
                        input_ids = input_ids[
                            :,
                            cache_position[-1]
                            - kwargs["selected_vokens"].shape[0]
                            - 2 :,
                        ]
                    if img_embeds_add is not None:
                        img_embeds_add = torch.cat(
                            [img_embeds_add, kwargs["selected_vokens"]], dim=-1
                        )
                    else:
                        img_embeds_add = kwargs["selected_vokens"].clone()
                else:
                    input_ids = input_ids[:, cache_position]
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "img_embeds_add": img_embeds_add,
            }
        )
        if "sub_image_masks" in kwargs:
            model_inputs.update(
                {
                    "sub_image_masks": kwargs["sub_image_masks"],
                }
            )
        return model_inputs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"
        return cache_name, past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # # update past_key_values keeping its naming used in model code
        # cache_name, cache = self._extract_past_from_model_output(outputs)
        # model_kwargs[cache_name] = cache
        ALL_CACHE_NAMES = [
            "past_key_values",  # default
            "cache_params",  # mamba-based models
            "state",  # rwkv
            "mems",  # xlnet
            "past_buckets_states",  # reformer
        ]
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "selected_vokens" in outputs and outputs["selected_vokens"] is not None:
            model_kwargs["selected_vokens"] = outputs["selected_vokens"]
        elif "selected_vokens" in model_kwargs:
            model_kwargs.pop("selected_vokens")

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )
        # raise Exception()
        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
                if (
                    "selected_vokens" in outputs
                    and outputs["selected_vokens"] is not None
                ):
                    model_kwargs["attention_mask"] = torch.cat(
                        [
                            model_kwargs["attention_mask"],
                            torch.ones_like(
                                outputs["selected_vokens"][:, 0].unsqueeze(0)
                            ),
                            torch.tensor(
                                [[1, 1]],
                                device=model_kwargs["attention_mask"].device,
                                dtype=model_kwargs["attention_mask"].dtype,
                            ),
                        ],
                        dim=-1,
                    )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )
        # TODO: cache_position is not applied to the prefix vokens
        if model_kwargs.get("use_cache", True):
            if "selected_vokens" in outputs and outputs["selected_vokens"] is not None:
                num_new_tokens += outputs["selected_vokens"].shape[0] + 2
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + num_new_tokens
            )
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1,
                past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype,
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs


num_selected_patches = 64
MCOT = config["MCOT"]


TRAING_CASE_1 = {
    "id": "physical-commonsense-1426",
    "category": "Status",
    "image_id": "commonsense-physical-commonsense-49",
    "question": "What general conclusion can you draw about this kitchen?",
    "choices": [
        "This is the kitchen of a restaurant",
        "The equipment in front has not been cleaned for a long time",
        "Someone searched in this kitchen",
        "All options are correct",
    ],
    "context": "",
    "answer": "D",
    "rationale": 'First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant.\nTherefore, option A is correct.\nSecond, there are grease stains on the front of appliances which are indicative of not being cleaned in a while.\nSo option B is correct answer.\nThird, cabinet doors are opened up throughout the kitchen which shows someone was searching for something.\nSo option C is incorrect.\nTherefore, we can infer that option A, B and C are all correct.\nSo, option "(D) All options are correct" is correct answer.',
    "split": "train",
    "image": "data\\images\\physical-commonsense-1426.png",
    "domain": "commonsense",
    "topic": "physical-commonsense",
}


dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
dataset = [x for x in dataset if x["image"] is not None]
# from datasets import load_dataset
# dataset = load_dataset("LightChen2333/M3CoT")['validation']

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForInterCoT.from_pretrained(
    model_path, torch_dtype="auto", attn_implementation=config["attn"]
).to("cuda:0")
# model_path = 'facebook/chameleon-7b'
# processor = ChameleonProcessor.from_pretrained(model_path)
# model = ChameleonForInterCoT.from_pretrained(model_path, attn_implementation=config['attn']).to(device='cuda', dtype=torch.bfloat16)

generation_config = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "min_new_tokens": 32,
    "max_new_tokens": 128,
}


def calculate_generated_text(prompt, vision_x, return_image_masks=False):
    """
    Calculate generated text given a prompt and vision data.

    Parameters:
    - prompt (str): The input prompt.
    - vision_x (list[PIL Images]): List of PIL Images containing vision data.

    Returns:
    Tuple[str, str]: Tuple containing the raw and salt answer text.
    """

    """
    Example Prompt:
    In zero-shot: "<image> <Question> <Options> Answer: "
    In few-shot: "<image> <Question> <Options> Answer: <Answer> <image> <Question> <Options> Answer: "
    """

    # # Preparation for inference
    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs, sub_image_masks = processor(
        text=[text],
        images=vision_x,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
        return_image_masks=True,
    )
    # inputs, sub_image_masks= processor(text = prompt, images=vision_x, padding=True, return_tensors="pt", return_for_text_completion=False, return_image_masks=True)

    inputs = inputs.to(device="cuda", dtype=torch.bfloat16)
    sub_image_masks = sub_image_masks.view(1, -1) if return_image_masks else None

    if return_image_masks:
        padding = torch.full(
            (1, 2048 - sub_image_masks.shape[-1]), True, dtype=torch.bool
        )
        sub_image_masks = (
            torch.cat([sub_image_masks, padding], dim=1).to(device="cuda").view(2, -1)
        )
        if prompt.count("<image>") == 2:  # two global images are provided
            sub_image_masks = torch.cat(
                [torch.ones(1, 1024).bool().to(device="cuda"), sub_image_masks], dim=0
            ).to(device="cuda")
        inputs["sub_image_masks"] = sub_image_masks

    inputs["output_attentions"] = MCOT

    out = model.generate(**inputs, **generation_config)  #
    # out_old = out[0]

    out = out[0][inputs["input_ids"].shape[1] :]

    generated_text = processor.decode(out, skip_special_tokens=True)
    # text_old = processor.decode(skip_special_tokens=True)

    return generated_text


# zero_shot_prompt_template = '''<image>Question: {}
# Options:
# '''
zero_shot_prompt_template = lambda question, options, image_path: [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "There is an image and a question about it. You need to have a deep thinking before answering, When you output the '.' The system will cut the most similar image for you to rethinking. Please answer the question based on the image.",
            },
        ],
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"{image_path}",
            },
            # {
            #     "type": "image",
            #     "image": f"{image_path}",
            # },
            {"type": "text", "text": f"Question: {question} Options:{options}"},
        ],
    }
]
mcot_induct = """<image>Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
<image11-21-0-11>First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. Therefore, option A is correct. <image4-10-23-29>Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. So option B is correct answer. <image21-32-1-9>Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.
Answer: {}""".format(
    TRAING_CASE_1["question"], *TRAING_CASE_1["choices"], TRAING_CASE_1["answer"]
)

model_name = "qwen_2_5_vl"


def main():
    import os

    output_dir = f"./results/{model_name}/{DATA_NAME}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mcot_one_fh = open(
        f"./results/{model_name}/{DATA_NAME}/{model_name}_mcot_one.json", "a"
    )
    mcot_zero_fh = open(
        f"./results/{model_name}/{DATA_NAME}/{model_name}_mcot_zero.json", "a"
    )
    for data in tqdm(dataset):
        options = ""
        for i, c in zip(["A", "B", "C", "D", "E", "F"], data["choices"]):
            options += "{}. {}\n".format(i, c)
        mcot_input_str = zero_shot_prompt_template(
            data["question"],
            options,
            os.path.join(
                IMG_FOLDER,
                data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
            ),
        )
        one_shot_vision = [
            Image.open(
                os.path.join(
                    "/mnt/hdd/tanhz/vqa_test/data/m3cot/images",
                    TRAING_CASE_1["id"] + ".png",
                )
            ),
            Image.open(
                os.path.join(
                    "/mnt/hdd/tanhz/vqa_test/data/m3cot/images",
                    TRAING_CASE_1["id"] + ".png",
                )
            ),
            Image.open(
                os.path.join(
                    IMG_FOLDER,
                    data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
                )
            ),
        ]

        zero_shot_vision = [
            Image.open(
                os.path.join(
                    IMG_FOLDER,
                    data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
                )
            )
        ]
        # zero_shot_vision = data['image']

        # one_shot_mcot_input_str = mcot_induct+'\n'+mcot_input_str

        zero_shot_mcot_input_str = mcot_input_str

        # "<image>Question: What general conclusion can you draw about this kitchen?\nOptions:\nA. This is the kitchen of a restaurant\nB. The equipment in front has not been cleaned for a long time\nC. Someone searched in this kitchen\nD. All options are correct\n<image11-21-0-11>First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. Therefore, option A is correct. <image4-10-23-29>Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. So option B is correct answer. <image21-32-1-9>Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.\nAnswer: D\n<image>Question: What is the likely purpose of the troll statue under the bridge?\nOptions:\nA. To scare away trespassers\nB. To bring attention to the city's tourist attractions\nC. To honor a local legend\nD. To discourage people from walking across the bridge\n"
        # one_shot = calculate_generated_text(one_shot_mcot_input_str, one_shot_vision, return_image_masks= True)
        zero_shot = calculate_generated_text(
            zero_shot_mcot_input_str, zero_shot_vision, return_image_masks=False
        )

        # oneshot_mcot_output = copy.deepcopy(data)
        # oneshot_mcot_output['pred'] = one_shot
        # oneshot_mcot_output['pred_old'] = one_old

        zeroshot_mcot_output = copy.deepcopy(data)
        zeroshot_mcot_output["pred"] = zero_shot

        # mcot_one_fh.write(json.dumps(oneshot_mcot_output) + '\n')
        mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + "\n")


def test():
    # 仿照main函数，但是只测10个样例
    import os

    output_dir = f"./results/{model_name}/{DATA_NAME}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mcot_one_fh = open(
        f"./results/{model_name}/{DATA_NAME}/{model_name}_mcot_one_test.json", "a"
    )
    mcot_zero_fh = open(
        f"./results/{model_name}/{DATA_NAME}/{model_name}_mcot_zero_test.json", "a"
    )
    for data in tqdm(dataset[:10]):
        options = ""
        for i, c in zip(["A", "B", "C", "D", "E", "F"], data["choices"]):
            options += "{}. {}\n".format(i, c)
        mcot_input_str = zero_shot_prompt_template(
            data["question"],
            options,
            os.path.join(
                IMG_FOLDER,
                data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
            ),
        )

        one_shot_vision = [
            Image.open(
                os.path.join(
                    "/mnt/hdd/tanhz/vqa_test/data/m3cot/images",
                    TRAING_CASE_1["id"] + ".png",
                )
            ),
            Image.open(
                os.path.join(
                    "/mnt/hdd/tanhz/vqa_test/data/m3cot/images",
                    TRAING_CASE_1["id"] + ".png",
                )
            ),
            Image.open(
                os.path.join(
                    IMG_FOLDER,
                    data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
                )
            ),
        ]

        zero_shot_vision = [
            Image.open(
                os.path.join(
                    IMG_FOLDER,
                    data["id"] + ".png" if DATA_NAME == "m3cot" else data["image"],
                )
            )
        ]
        # zero_shot_vision = data['image']

        # one_shot_mcot_input_str = mcot_induct+'\n'+mcot_input_str

        zero_shot_mcot_input_str = mcot_input_str

        # one_shot = calculate_generated_text(one_shot_mcot_input_str, one_shot_vision, return_image_masks= True)
        # try:
        zero_shot = calculate_generated_text(
            zero_shot_mcot_input_str, zero_shot_vision, return_image_masks=False
        )
        # except Exception as e:
        # zero_shot = f'error:{e}'

        # oneshot_mcot_output = copy.deepcopy(data)
        # oneshot_mcot_output['pred'] = one_shot
        # oneshot_mcot_output['pred_old'] = one_old

        zeroshot_mcot_output = copy.deepcopy(data)
        zeroshot_mcot_output["pred"] = zero_shot

        # mcot_one_fh.write(json.dumps(oneshot_mcot_output) + '\n')
        mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + "\n")


if __name__ == "__main__":
    # test()
    main()


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]
# image = Image.open('/mnt/hdd/tanhz/vqa_test/data/m3cot/train/images/physical-commonsense-1424.png')

# text  = """<image>Question:
# Options:
# <image11-21-0-11>First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. Therefore, option A is correct. <image4-10-23-29>Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. So option B is correct answer. <image21-32-1-9>Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.
# """
# inputs, sub_image_masks = processor(images=image, text=text, padding=True, return_tensors="pt",return_image_masks=True)
# # # Preparation for inference
# # text = processor.apply_chat_template(
# #     messages, tokenize=False, add_generation_prompt=True
# # )
# # image_inputs, video_inputs = process_vision_info(messages)
# # inputs = processor(
# #     text=[text],
# #     images=image_inputs,
# #     videos=video_inputs,
# #     padding=True,
# #     return_tensors="pt",
# # )
# inputs = inputs.to("cuda:0")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
