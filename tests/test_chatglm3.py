#!/usr/bin/env python3
import sys
import os
import time
import json
import torch
import numpy as np
import lightning as L
from pathlib import Path
from lit_gpt.model import GPT, Block, Config
from torch.nn.utils import skip_init
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    lazy_load,
    num_parameters,
)
from lit_gpt.tokenizer import Tokenizer
from tests.test_chatglm2 import run_lit, run_hf, compare_diff
from transformers import AutoTokenizer
import glob
import contextlib
import gc


def model_diff(hf_root):

    sys.path.append(hf_root)
    import modeling_chatglm, configuration_chatglm

    # load lit-gpt model
    st = time.time()
    config_name = os.path.basename(hf_root)
    config = Config.from_name(config_name)
    fabric = L.Fabric(devices=1, precision="16-true")
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    # checkpoint_dir = Path("./checkpoints/chatglm/chatglm3-6b-hf")
    checkpoint_dir = Path(f"./checkpoints/chatglm/{config_name}")
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    # with lazy_load(checkpoint_path) as checkpoint:
    checkpoint = lazy_load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"load lit-gpt model cost: {time.time() - st:.2f}s")

    # load hf model
    hf_load_st = time.time()
    file = f"{hf_root}/config.json"
    with open(file, 'r') as f:
        cfgs = json.load(f)
    hf_config = configuration_chatglm.ChatGLMConfig(**cfgs)
    hf_model = modeling_chatglm.ChatGLMForConditionalGeneration(hf_config)

    bin_files = glob.glob(f"{checkpoint_dir}/*.bin")
    hf_state_dict = {}
    # with contextlib.ExitStack() as stack:
    for bin_file in sorted(bin_files):
        print("Processing", bin_file)
        hf_weights = lazy_load(bin_file)
        hf_state_dict.update(hf_weights)
    hf_model.load_state_dict(hf_state_dict)
    hf_model = hf_model.to("cuda:1")
    print(f"load hf model cost: {time.time() - hf_load_st:.2f}s")

    token_ids = [53456, 33071, 32884, 31123,  5091,   383,   344, 21590]
    input_ids = torch.tensor([token_ids])

    lit_out = run_lit(input_ids, model, config)
    hf_out = run_hf(input_ids, hf_model, hf_config)
    for k, v in lit_out.items():
        print(k, v.shape, hf_out[k].shape, v.dtype, hf_out[k].dtype)
        diff = compare_diff(v, hf_out[k], k)
        print(f"{k} diff: {diff:.6f}")

    print(f"all done, total cost: {time.time() - st:.2f}s")


def lit_params(config_name="chatglm3-6b-hf"):
    config = Config.from_name(config_name)
    fabric = L.Fabric(devices=1, precision="bf16-mixed")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        for name, para in model.named_parameters():
            print(f'{name}: {para.size()}')


def test_load_lit(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric = L.Fabric(devices=1, precision="16-mixed")
    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    # with lazy_load(checkpoint_path) as checkpoint:
    checkpoint = lazy_load(checkpoint_path)
    model.load_state_dict(checkpoint)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")


def test_tokenizer(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric = L.Fabric(devices=1, precision="16-mixed")
    tokenizer = Tokenizer(checkpoint_dir)
    prompt = "今天天气不错，how are you?."

    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    # encoded = hf_tokenizer.build_chat_input(prompt, history=[])
    print('huggingface prefix_tokens: ', hf_tokenizer.get_prefix_tokens())

    encoded = tokenizer.encode(prompt, device=fabric.device)
    print(f"lit-gpt encode: {encoded}")
    decode = tokenizer.decode(encoded)
    print(f'lit-gpt decode: {decode}')
    print(f'lit-gpt bos_id, eos_id: {tokenizer.bos_id}, {tokenizer.eos_id}')

    print(f"huggingface prompt: {prompt}")
    inputs = hf_tokenizer([prompt], return_tensors="pt")
    print(f"huggingface input: {inputs}")
    hf_encoded = hf_tokenizer.encode(prompt)
    print(f"huggingface encode: {hf_encoded}")
    hf_decoded = hf_tokenizer.decode(hf_encoded)
    print(f'huggingface decode: {hf_decoded}')
    print(f'huggingface bos eos: {hf_tokenizer.bos_token_id}, {hf_tokenizer.eos_token_id}, {hf_tokenizer.pad_token_id}')
    print(f"-- huggingface decoded details: ")
    for i in hf_encoded:
        print(f'{i}: {hf_tokenizer._convert_id_to_token(i)}')



if __name__ == "__main__":
    import fire
    fire.Fire()

# vim: ts=4 sw=4 sts=4 expandtab
