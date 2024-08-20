import os
import sys
import math
import torch
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig

from llavavid.model import LlavaLlamaForCausalLM
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
from llavavid.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

from mv_dts import MusicVideoDatasetV3

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import deepspeed


def parse_args(args=None):
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--test_portion", type=float, default=1.0)
    parser.add_argument("--offset_ratio", type=float, default=0.1)

    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default="")

    return parser.parse_args(args=args)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", overwrite_config=None):
    kwargs = {"device_map": device_map}

    # import pdb;pdb.set_trace()
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float16

   # this may be mm projector only
    print("Loading LLaVA from base model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    if overwrite_config is not None:
        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    image_processor = None

    assert "llava" in model_name.lower(), "Only LLaVA models are supported for video chatbot."
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    vision_tower.to(device=model.device, dtype=model.dtype)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def create_single_round_conv_labels(prompt, input_ids, tokenizer, conv):
    assert conv.sep_style == SeparatorStyle.TWO
    labels = input_ids.clone()
    # Mask targets
    assistant_seperator = conv.sep + conv.roles[1] + ": "
    parts = prompt.split(assistant_seperator)
    assert len(parts) == 2  # first part should be 'system' + 'user' prompt, while the second part should be 'assistant' + 'user' prompt

    parts[0] += assistant_seperator

    immune_length = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # ignore EOS token

    labels[:immune_length] = IGNORE_INDEX

    assert (len(tokenizer_image_token(parts[0], tokenizer)) + len(tokenizer_image_token(parts[1], tokenizer)) - 2) == labels.shape[0], "This assumption should be True"

    return labels


@torch.inference_mode()
def evaluation(model, enum_iter, device="cuda"):
    eval_preds = []
    eval_labels = []
    eval_outputs = []
    eval_miss_tokens = 0

    for i, batch in enum_iter:
        assert batch["frames"].shape[0] == 1

        t_pred, miss, output = model.generate(
            batch["frames"],
            batch["labels"],
            device=device
        )

        t_labels = batch["labels"].long()

        if (miss):
            eval_miss_tokens += 1

        eval_labels.extend(t_labels.tolist())
        eval_preds.append(t_pred)
        eval_outputs.append(output)

    return dict(
        eval_preds=eval_preds,
        eval_labels=eval_labels,
        eval_outputs=eval_outputs,
        eval_miss_tokens=eval_miss_tokens
    )


def main():
    seed_everything(1019)

    args = parse_args()

    CONV_MODE = "vicuna_v1"
    DURATION = args.duration
    FRAMES_NUM = args.frames
    TEST_PORTION = args.test_portion
    T_RES_CLS = args.segments
    MODEL_PATH = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
    BATCH_SIZE = args.batch_size
    OFFSET_RATIO = args.offset_ratio

    # this is weird, but env vars are much detailed than args...
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
    GLOBAL_RANK = int(os.getenv('RANK', '0'))

    DS_CONFIG = {
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 0
        },
        "local_rank": LOCAL_RANK
    }

    # create model
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # load pretrained model and auxiliary components
            self.tokenizer, self.model, self.image_processor, self.context_len = self.load_pretrained_model()
            self.model.half().requires_grad_(False)

            question = (
                DEFAULT_IMAGE_TOKEN + "\n" +
                f"If the given video was split into {T_RES_CLS-1} equal duration segments, focusing on the feelings and atmosphere from the visuals of each segment, tell me which segment most possibly indicate a shift in atmosphere and why? " +
                f"Give me a short answer. The options should be: " + ', '.join([f'Segment {i+1}.' for i in range(T_RES_CLS - 1)]) + "or 'None.'."
            )

            # prepare testing inputs
            self.test_input_ids, _, self.test_attention_masks, stop_str = self.make_templates(question, "", self.tokenizer)
            self.stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, self.test_input_ids)

        def load_pretrained_model(self):
            # Initialize the model
            model_name = get_model_name_from_path(MODEL_PATH)

            # Set model configuration parameters if they exist
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = "spatial_pool"
            overwrite_config["mm_spatial_pool_stride"] = 2
            overwrite_config["mm_spatial_pool_out_channels"] = 1024
            overwrite_config["mm_spatial_pool_mode"] = "average"
            overwrite_config["patchify_video_feature"] = False

            cfg_pretrained = AutoConfig.from_pretrained(MODEL_PATH)

            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = FRAMES_NUM * (16 // 2)**2 + 1000
            else:
                least_token_number = FRAMES_NUM * (24 // 2)**2 + 1000

            scaling_factor = math.ceil(least_token_number / 4096)

            if scaling_factor >= 2:
                if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            # Load model with new configuration
            return load_pretrained_model(
                MODEL_PATH,
                None,
                model_name,
                False,
                overwrite_config=overwrite_config,
                device_map="cpu"
            )

        def make_templates(self, question, answer, tokenizer):
            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

            if (answer == ""):
                labels = input_ids.clone()
                labels[:] = IGNORE_INDEX
            else:
                labels = create_single_round_conv_labels(prompt, input_ids, tokenizer, conv)

            labels = labels[None].cpu()
            input_ids = input_ids[None].cpu()
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cpu()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            return input_ids, labels, attention_masks, stop_str

        def generate(self, frames, t_labels, device="cuda"):
            b = frames.shape[0]

            assert b == 1

            frames = torch.stack([
                model.image_processor.preprocess(frame, return_tensors="pt")["pixel_values"]
                for frame in frames
            ]).to(
                device=device,
                dtype=self.model.dtype
            )

            t_labels = t_labels.to(device=device)

            results = self.model.generate(
                inputs=self.test_input_ids.to(device=device),
                images=frames.to(device=device),
                attention_mask=self.test_attention_masks.to(device=device),
                modalities="video",
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[self.stopping_criteria],
                output_hidden_states=False,
                return_dict_in_generate=True
            )

            output_ids = results["sequences"]

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()

            label = -1
            miss = False

            for i in range(1, T_RES_CLS):
                if (f"segment {i}." in outputs):
                    label = i
                    break
            else:
                label = T_RES_CLS
                miss = True

            return label - 1, miss, outputs

    model = Model()

    # intialize deepspeed
    model_engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model,
        model_parameters=[]
    )

    assert type(model_engine) == deepspeed.runtime.engine.DeepSpeedEngine, "we assume the specified engine for further dataloader creation"

    def _single_shot_eval():
        test_loader = model_engine.deepspeed_io(
            MusicVideoDatasetV3(
                "test",
                frames=FRAMES_NUM,
                t_res=T_RES_CLS,
                duration=DURATION,
                portion=TEST_PORTION,
                offset_ratio=OFFSET_RATIO,
                reverse=True
            ),
            route=deepspeed.runtime.constants.ROUTE_EVAL
        )

        pbar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            ncols=0,
            disable=(not (LOCAL_RANK == 0))
        )

        model_engine.module.eval()
        eval_results = evaluation(
            model=model_engine.module,
            enum_iter=pbar,
            device=model_engine.device
        )
        model_engine.module.train()

        # gather all results from distributions, for metrics computation.
        gathered_results = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            gathered_results,
            eval_results
        )
        torch.distributed.barrier()

        # compute & log metrics only on RANK 0.
        if (GLOBAL_RANK == 0):
            eval_labels = np.concatenate([results["eval_labels"] for results in gathered_results], axis=0)
            eval_preds = np.concatenate([results["eval_preds"] for results in gathered_results], axis=0)
            eval_miss_tokens = sum([results["eval_miss_tokens"] for results in gathered_results])
            eval_outputs = [output for results in gathered_results for output in results["eval_outputs"]]
            eval_accuracy = (eval_labels == eval_preds).astype(np.float32).mean()

            events = [
                ("Eval/Accuracy", eval_accuracy, model_engine.global_samples),
                ("Eval/Miss", eval_miss_tokens, model_engine.global_samples),
            ]

            for event in events:
                print(event[0], event[1])

            with open("result.pickle", "wb") as f:
                pickle.dump(
                    dict(
                        eval_labels=eval_labels,
                        eval_preds=eval_preds,
                        eval_outputs=eval_outputs,
                        eval_miss_tokens=eval_miss_tokens
                    ),
                    f
                )

    _single_shot_eval()


if __name__ == "__main__":
    main()
