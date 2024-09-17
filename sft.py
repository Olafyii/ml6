# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

# python sft.py --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" --output_dir="test" --data_json_path test.json --dataset_text_field="text" --report_to="wandb" --learning_rate=1.41e-5 --per_device_train_batch_size=64 --gradient_accumulation_steps=16 --output_dir="test" --logging_steps=1 --num_train_epochs=3 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=64 --lora_alpha=16 --run_name "test" --wandb_project "webarena"
import logging
import os
import json
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    import wandb
    from typing import Optional
    from dataclasses import dataclass, field

    @dataclass
    class CustomArguments:
        data_json_path: Optional[str] = field(
            default='',
            metadata={"help": "Path to the JSON file containing data."}
        )
        wandb_project: Optional[str] = field(
            default='webarena',
            metadata={"help": "WandB project name."}
        )
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, CustomArguments)) 
    args, training_args, model_config, custom_args = parser.parse_args_and_config()

    ################
    # FSDP
    ################
    # config = {
    #     "compute_environment": "LOCAL_MACHINE",
    #     "debug": False,
    #     "distributed_type": "FSDP",
    #     "downcast_bf16": "no",
    #     "fsdp_config": {
    #         "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    #         "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
    #         "fsdp_cpu_ram_efficient_loading": True,
    #         "fsdp_forward_prefetch": False,
    #         "fsdp_offload_params": True,
    #         "fsdp_sharding_strategy": 1,
    #         "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    #         "fsdp_sync_module_states": True,
    #         "fsdp_transformer_layer_cls_to_wrap": "BertLayer",
    #         "fsdp_use_orig_params": True
    #     },
    #     "machine_rank": 0,
    #     "main_training_function": "main",
    #     "mixed_precision": "bf16",
    #     "num_machines": 1,
    #     "num_processes": 2,
    #     "rdzv_backend": "static",
    #     "same_network": True,
    #     "tpu_env": [],
    #     "tpu_use_cluster": False,
    #     "tpu_use_sudo": False,
    #     "use_cpu": False
    # }
    # training_args.fsdp_config = config

    ################

    os.environ["WANDB_PROJECT"] = custom_args.wandb_project
    # wandb.init(project=custom_args.wandb_project, name=training_args.run_name)
    # wandb.config.update(custom_args)

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # raw_datasets = load_dataset(args.dataset_name)

    # train_dataset = raw_datasets[args.dataset_train_split]
    # eval_dataset = raw_datasets[args.dataset_test_split]

    data_json = json.load(open(custom_args.data_json_path))
    data_json.update({training_args.dataset_text_field if training_args.dataset_text_field else 'text': \
                      [prompt + completion for prompt, completion in zip(data_json['prompt'], data_json['completion'])]})
    train_dataset = Dataset.from_dict(data_json)
    eval_dataset = None

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)

#  WANDB_MODE=disabled accelerate launch sft.py --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" --dataset_text_field="text" --report_to="wandb" --learning_rate=3e-4 --gradient_accumulation_steps=1 --logging_steps=1 --num_train_epochs=3 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=64 --lora_alpha=16 --run_name "test" --wandb_project "ml6_test" --data_json_path="test.json" --output_dir="test" --per_device_train_batch_size 4 --fp16