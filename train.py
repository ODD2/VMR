import os
import sys
import math
import torch
import random
import pickle
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from functools import partial
from datetime import datetime
from contextlib import contextmanager

from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig

from llavavid.model import LlavaLlamaForCausalLM
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
from llavavid.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

from mv_dts import MusicVideoDataset

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import deepspeed


def parse_args(args=None):
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--step_batch", type=int, default=20)
    parser.add_argument("--eval_step", type=float, default=50)

    # from checkpoints or pretrained weights
    parser.add_argument("--from_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_tag", type=str, default=None)

    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--train_portion", type=float, default=1.0)
    parser.add_argument("--test_portion", type=float, default=1.0)
    parser.add_argument("--offset_ratio", type=float, default=0.1)

    parser.add_argument("--peft_mode", type=str, default="qkv")
    parser.add_argument("--lora_rank", type=int, default=16)

    parser.add_argument("--proj_mode", type=str, default="v1")
    parser.add_argument("--inference", action="store_true")

    parser.add_argument("--dts_ver", type=str, default="v1")

    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default="")

    return parser.parse_args(args=args)


@contextmanager
def _module_init(speed_skip):
    if (speed_skip):
        kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        uniform_ = torch.nn.init.uniform_
        normal_ = torch.nn.init.normal_
        torch.nn.init.normal_ = lambda x, *args, **kargs: x
        torch.nn.init.kaiming_uniform_ = lambda x, *args, **kargs: x
        torch.nn.init.uniform_ = lambda x, *args, **kargs: x
        yield
        torch.nn.init.kaiming_uniform_ = kaiming_uniform_
        torch.nn.init.uniform_ = uniform_
        torch.nn.init.normal_ = normal_
    else:
        pass
        yield
        pass


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    overwrite_config=None
):
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


class TSProj(torch.nn.Module):
    def __init__(self, classes=2, mode="v1"):
        super(TSProj, self).__init__()
        self.mode = mode

        if (self.mode == "v1"):
            self.proj1 = torch.nn.Linear(4096, 4096, bias=False)
            torch.nn.init.zeros_(self.proj1.weight)
            self.gelu1 = torch.nn.GELU()
            self.proj2 = torch.nn.Linear(4096, 4096, bias=False)
            self.gelu2 = torch.nn.GELU()
            torch.nn.init.zeros_(self.proj2.weight)
            self.proj3 = torch.nn.Linear(4096, classes, bias=False)

        elif (self.mode == "v2"):
            self.proj3 = torch.nn.Linear(4096, classes, bias=False)

        else:
            raise NotImplementedError()

    def forward(self, x):
        if (self.mode == "v1"):
            x = self.gelu1((x + self.proj1(x)))
            x = self.gelu2((x + self.proj2(x)))
            return self.proj3(x)

        elif (self.mode == "v2"):
            x = self.proj3(x)
            return x


class VicunaLLaVANeXTVideoDPO(torch.nn.Module):
    def __init__(self, frames, segments, proj_mode, peft_mode, lora_rank):
        super(VicunaLLaVANeXTVideoDPO, self).__init__()
        self.conv_mode = "vicuna_v1"
        self.model_path = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
        self.segments = segments
        self.frames = frames

        # load pretrained model and auxiliary components
        self.tokenizer, self.model, self.image_processor, self.context_len = self.build_pretrained_components()
        self.model.half().requires_grad_(False)

        # hook peft modules
        self.model = self.make_peft(self.model, peft_mode, lora_rank)

        # add special token into tokenizer
        self.tokenizer.add_tokens("[TS]")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.SEG_TOKEN_IDX = self.tokenizer("[TS]", add_special_tokens=False).input_ids[-1]

        question = (
            DEFAULT_IMAGE_TOKEN + "\n" +
            f"If the given video was split into {segments-1} equal duration segments, focusing on the feelings and atmosphere from the visuals of each segment, tell me which segment most possibly indicate a shift in atmosphere and why? Give me a short answer."
        )
        answer = "It's segment [TS]."

        # prepare training inputs
        self.input_ids, self.labels, self.attention_masks, _ = self.make_templates(question, answer, self.tokenizer)
        # prepare testing inputs
        self.test_input_ids, _, self.test_attention_masks, stop_str = self.make_templates(question, "", self.tokenizer)
        self.stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, self.test_input_ids)

        # prepare temporal classification projection layer
        self.t_proj = TSProj(
            classes=segments,
            mode=proj_mode
        ).half()

        # extend trainable parameters
        for n, p in self.model.named_parameters():
            if any([x in n for x in ["lm_head", "embed_tokens"]]):
                print(f"Turn LLM Param '{n}({p.shape})' into Trainable.")
                p.requires_grad = True
        self.t_proj.requires_grad_(True)

    def tokenizer_image_token(self, prompt):
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        offset = 0
        input_ids = []
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)

    def create_single_round_conv_labels(self, prompt, input_ids, conv):
        assert conv.sep_style == SeparatorStyle.TWO
        labels = input_ids.clone()
        # Mask targets
        assistant_seperator = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(assistant_seperator)
        assert len(parts) == 2  # first part should be 'system' + 'user' part, while the second part should be the 'assisant' part.

        parts[0] += assistant_seperator

        immune_length = len(self.tokenizer_image_token(parts[0])) - 1  # ignore EOS token

        labels[:immune_length] = IGNORE_INDEX

        assert (len(self.tokenizer_image_token(parts[0])) + len(self.tokenizer_image_token(parts[1])) - 2) == labels.shape[0], "This assumption should be True"

        return labels

    def build_pretrained_components(self):
        # Initialize the model
        model_name = get_model_name_from_path(self.model_path)

        # Set model configuration parameters if they exist
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = "spatial_pool"
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_out_channels"] = 1024
        overwrite_config["mm_spatial_pool_mode"] = "average"
        overwrite_config["patchify_video_feature"] = False

        cfg_pretrained = AutoConfig.from_pretrained(self.model_path)

        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = self.frames * (16 // 2)**2 + 1000
        else:
            least_token_number = self.frames * (24 // 2)**2 + 1000

        scaling_factor = math.ceil(least_token_number / 4096)

        if scaling_factor >= 2:
            if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                print(float(scaling_factor))
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        # Load model with new configuration
        return load_pretrained_model(
            self.model_path,
            None,
            model_name,
            False,
            overwrite_config=overwrite_config,
            device_map="cpu"
        )

    def make_peft(self, model, peft_mode, lora_rank):
        # initialize peft model
        linear_names = find_all_linear_names(model)

        if peft_mode == "qkv":
            linear_names = [n for n in linear_names if "q_proj" in n or "k_proj" in n or "v_proj" in n]
        elif peft_mode == "q0":
            linear_names = sorted([n for n in linear_names if "q_proj" in n])[:1]
        elif peft_mode == "all":
            pass
        else:
            raise NotImplementedError()

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=linear_names,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        return model

    def make_templates(self, question, answer, tokenizer):
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt)

        if (answer == ""):
            labels = input_ids.clone()
            labels[:] = IGNORE_INDEX
        else:
            labels = self.create_single_round_conv_labels(prompt, input_ids, conv)

        labels = labels[None].cpu()
        input_ids = input_ids[None].cpu()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cpu()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        return input_ids, labels, attention_masks, stop_str

    def __prepare_inputs_labels_for_multimodal(self, frames, device="cuda"):
        b = frames.shape[0]

        frames = torch.stack([
            self.image_processor.preprocess(
                frame,
                return_tensors="pt"
            )["pixel_values"]
            for frame in frames
        ]).to(
            device=device,
            dtype=self.model.dtype
        )

        (_, _, _attention_masks, _, _inputs_embeds, _labels) = self.model.prepare_inputs_labels_for_multimodal(
            self.input_ids.repeat(b, 1).to(device),
            None,
            self.attention_masks.repeat(b, 1).to(device),
            None,
            self.labels.repeat(b, 1).to(device),
            frames,
            "videos",
            None
        )

        return _attention_masks, _inputs_embeds, _labels

    def forward(self, frames, t_labels, device="cuda"):
        b = frames.shape[0]

        (_attention_masks, _inputs_embeds, _labels) = self.__prepare_inputs_labels_for_multimodal(frames, device)

        t_labels = t_labels.to(device=device)

        results = self.model(
            input_ids=None,
            position_ids=None,
            attention_mask=_attention_masks,
            past_key_values=None,
            inputs_embeds=_inputs_embeds,
            labels=_labels,
            cache_position=None,
            return_dict=None,
            output_hidden_states=True,
            output_attentions=False
        )

        seg_token_mask = torch.cat(
            [
                _labels[:, 1:].eq(self.SEG_TOKEN_IDX),
                torch.zeros((b, 1)).bool().to(device)
            ],
            dim=1
        )

        embeds = results["hidden_states"][-1][seg_token_mask]

        t_logits = self.t_proj(embeds).float()

        loss_c = torch.nn.functional.cross_entropy(t_logits, t_labels)

        loss_d = results["loss"]

        return (loss_c, loss_d)

    def generate(self, frames, t_labels, device="cuda"):
        b = frames.shape[0]

        assert b == 1

        frames = torch.stack([
            self.image_processor.preprocess(
                frame,
                return_tensors="pt"
            )["pixel_values"]
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
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        output_hiddens = results["hidden_states"]
        output_ids = results["sequences"]

        miss = False
        for i, token_id in enumerate(output_ids[0]):
            if token_id == self.SEG_TOKEN_IDX:
                ts_embed = output_hiddens[i][-1][0]
                t_probs = self.t_proj(ts_embed).float().softmax(dim=-1)
                break
        else:
            t_probs = torch.ones((b, self.segments)) / self.segments
            miss = True

        return t_probs, miss


def main():
    seed_everything(1019)

    args = parse_args()

    run_id = datetime.now().strftime('%m%d_%H%M%S')

    args.base_log_dir = f"./logs/{run_id}/"

    assert (not args.inference) or (not args.from_ckpt is None), "Inference mode requires a checkpoint to load."

    # fetch distributed environment variables for deepspeed.
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', '0'))
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
    GLOBAL_RANK = int(os.getenv('RANK', '0'))

    DS_CONFIG = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.step_batch,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.99],
                "eps": 1e-4,
                "weight_decay": 0
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100
            }
        },
        "gradient_clipping": 1.0,
        "wandb": {
            "enabled": not args.inference,
            "project": "VideoMusic"
        },
        "local_rank": LOCAL_RANK
    }

    # create model
    with _module_init(speed_skip=(not args.from_ckpt is None)):
        model = VicunaLLaVANeXTVideoDPO(
            frames=args.frames,
            segments=args.segments,
            proj_mode=args.proj_mode,
            peft_mode=args.peft_mode,
            lora_rank=args.lora_rank
        )

    model_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    print(f"Trainable Parameters: {list(model_params.keys())}")

    # intialize deepspeed
    model_engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model,
        model_parameters=list(model_params.values())
    )

    if args.from_ckpt is not None:
        model_engine.load_checkpoint(
            args.from_ckpt,
            tag=args.ckpt_tag,
            load_module_strict=False,
            load_optimizer_states=not args.inference,
            load_lr_scheduler_states=not args.inference,
            load_module_only=args.inference,
            custom_load_fn=None
        )

    assert type(model_engine) == deepspeed.runtime.engine.DeepSpeedEngine, "we assume the specified engine for further dataloader creation"

    # sync logging folder location
    gathered_results = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        gathered_results,
        args.base_log_dir
    )
    torch.distributed.barrier()
    args.base_log_dir = gathered_results[0]

    # save run command
    if (GLOBAL_RANK == 0 and not args.inference):
        os.makedirs(args.base_log_dir, exist_ok=True)
        with open(os.path.join(args.base_log_dir, "cmd.sh"), "w") as f:
            f.write(' '.join(sys.argv))

    def _run_evaluation_loop(in_training=False):
        test_loader = model_engine.deepspeed_io(
            MusicVideoDataset(
                "test",
                frames=args.frames,
                t_res=args.segments,
                duration=args.duration,
                portion=args.test_portion,
                offset_ratio=args.offset_ratio,
                reverse=True,
                dts_ver=args.dts_ver
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
        # evaluation logic
        with torch.inference_mode():
            eval_probs = []
            eval_labels = []
            eval_miss_tokens = 0

            for i, batch in pbar:
                assert batch["frames"].shape[0] == 1

                t_probs, miss = model.generate(
                    batch["frames"],
                    batch["labels"],
                    device=model_engine.device
                )

                t_labels = batch["labels"].long()

                if (miss):
                    eval_miss_tokens += 1

                assert len(t_labels.shape) == 1
                assert len(t_probs.shape) == 2

                eval_labels.extend(t_labels.tolist())
                eval_probs.extend(t_probs.tolist())

            eval_results = dict(
                eval_probs=eval_probs,
                eval_labels=eval_labels,
                eval_miss_tokens=eval_miss_tokens
            )

        model_engine.module.train()

        if in_training:
            # checkpointing
            model_engine.save_checkpoint(args.base_log_dir, exclude_frozen_parameters=True, tag=f"global_step{model_engine.global_samples}")

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
            eval_probs = np.concatenate([results["eval_probs"] for results in gathered_results], axis=0)
            eval_miss_tokens = sum([results["eval_miss_tokens"] for results in gathered_results])

            eval_preds = eval_probs.argmax(axis=-1)
            eval_accuracy = (eval_labels == eval_preds).astype(np.float32).mean()
            eval_loss = -np.log(eval_probs[range(len(eval_labels)), eval_labels] + 1e-4).mean()

            events = [
                ("Eval/Accuracy", eval_accuracy, model_engine.global_samples),
                ("Eval/Loss", eval_loss, model_engine.global_samples),
                ("Eval/Miss", eval_miss_tokens, model_engine.global_samples),
            ]

            if in_training:
                model_engine.monitor.write_events(events)
            else:
                for event in events:
                    print(event[0], event[1])

    def _run_training_loop():
        # begin training pipeline
        train_loader = model_engine.deepspeed_io(
            MusicVideoDataset(
                "train",
                frames=args.frames,
                t_res=args.segments,
                duration=args.duration,
                portion=args.train_portion,
                offset_ratio=args.offset_ratio,
                dts_ver=args.dts_ver
            )
        )

        for epoch in range(args.epochs):
            # train phase
            model_engine.module.train()

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                ncols=0,
                disable=(not (LOCAL_RANK == 0))
            )

            for i, batch in pbar:
                loss_c, loss_d = model_engine(batch["frames"], batch["labels"], device=model_engine.device)

                loss = loss_c + loss_d

                model_engine.backward(loss)
                model_engine.step()

                pbar.set_description(f"Loss: {loss:.2f} ({loss_d:.2f},{loss_c:.2f}), Epoch {epoch+1}")

                # eval phase
                if ((epoch * len(train_loader) + i + 1) % (args.step_batch * args.eval_step) == 0):
                    _run_evaluation_loop(in_training=True)

            model_engine.monitor.write_events(
                [
                    ("Train/Samples/Epochs", epoch + 1, model_engine.global_samples),
                ]
            )

    if args.inference:
        _run_evaluation_loop(in_training=False)
    else:
        _run_training_loop()


if __name__ == "__main__":
    main()
