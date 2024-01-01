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
from transformers import AutoTokenizer
import glob
import contextlib
import gc


def model_diff(hf_root):

    sys.path.append(hf_root)
    import modeling_baichuan, configuration_baichuan

    # load lit-gpt model
    st = time.time()
    config = Config.from_name("baichuan2-7b-chat-hf")
    fabric = L.Fabric(devices=1, precision="32-true")
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    checkpoint_dir = Path("./checkpoints/baichuan/baichuan2-7b-chat-hf")
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint = lazy_load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"load lit-gpt model cost: {time.time() - st:.2f}s")

    # load hf model
    hf_load_st = time.time()
    file = f"{hf_root}/config.json"
    with open(file, 'r') as f:
        cfgs = json.load(f)
    hf_config = configuration_baichuan.BaichuanConfig(**cfgs)
    hf_model = modeling_baichuan.BaichuanForCausalLM(hf_config)

    bin_files = glob.glob(f"{checkpoint_dir}/*.bin")
    hf_state_dict = {}
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
        # print(k, v.shape, hf_out[k].shape, v.dtype, hf_out[k].dtype)
        diff = compare_diff(v, hf_out[k], k)
        print(f"{k} diff: {diff:.6f}")

    print(f"all done, total cost: {time.time() - st:.2f}s")


def compare_diff(x, y, key=None):
    x = x.cpu().detach().numpy()
    y = y.reshape(x.shape)
    y = y.cpu().detach().numpy()
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    diff = np.abs(x - y).max()
    return diff


def run_lit(inputs, model, config):
    idx = inputs.to("cuda:0")
    B, T = idx.size()

    emb = model.transformer.wte(idx)

    out = model(idx)
    block = model.transformer.h[0]
    norm_1 = block.norm_1(emb)

    cos, sin = model.cos[:T].to(idx.device), model.sin[:T].to(idx.device)

    block_1, *_ = block(norm_1, cos, sin)
    return {
        "norm_1": norm_1,
        "emb": emb,
        "out": out,
        "block_1": block_1
    }


def run_hf(inputs, hf_model, config):
    idx = inputs.to("cuda:1")
    B, T = idx.size()
    emb = hf_model.model.embed_tokens(idx)

    y = hf_model.model(idx)[0]
    out = hf_model.lm_head(y)

    block = hf_model.model.layers[0]
    norm_1 = block.input_layernorm(emb)
    batch_size, seq_length = emb.shape[:2]
    past_key_values_length = 0
    attention_mask = torch.ones(
            (B, T), dtype=torch.bool, device=emb.device
        )
    attention_mask = hf_model.model._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), emb, past_key_values_length
    )

    block_1, *_ = block(norm_1, attention_mask=attention_mask)
    return {"emb": emb,
            "norm_1": norm_1,
            "out": out,
            "block_1": block_1
    }


def hf_params(hf_root):

    sys.path.append(hf_root)
    import modeling_baichuan, configuration_baichuan

    file = f"{hf_root}/config.json"
    with open(file, 'r') as f:
        cfgs = json.load(f)
    config = configuration_baichuan.BaichuanConfig(**cfgs)
    model = modeling_baichuan.BaichuanForCausalLM(config)
    for name, para in model.named_parameters():
        print(f'{name}: {para.size()}')


def lit_params():
    config = Config.from_name("baichuan2-7b-chat-hf")
    fabric = L.Fabric(devices=1, precision="bf16-true")
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        for name, para in model.named_parameters():
            print(f'{name}: {para.size()}')


def test_tokenizer(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric = L.Fabric(devices=1, precision="bf16-true")
    tokenizer = Tokenizer(checkpoint_dir)
    prompt = "今天天气不错，how are you?."
    encoded = tokenizer.encode(prompt, device=fabric.device)
    print(f"lit-gpt encoded: {encoded}")
    decode = tokenizer.decode(encoded)
    print(f'lit-gpt decode: {decode}')
    print(f'lit-gpt decode 0,1,2: {tokenizer.decode(np.array([0,1,2]))}')
    print(f'lit-gpt bos_id, eos_id, pad_id: {tokenizer.bos_id}, {tokenizer.eos_id}, {tokenizer.pad_id}')

    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    print('-- special tokens --')
    for k, v in hf_tokenizer.special_tokens_map.items():
        print('k, v, id: ', k, v, hf_tokenizer.convert_tokens_to_ids(v))
    hf_encoded = hf_tokenizer.encode(prompt)
    print(f"huggingface encoded: {hf_encoded}")
    hf_decoded = hf_tokenizer.decode(hf_encoded)
    print(f'huggingface decoded: {hf_decoded}')
    print(f"huggingface decoded details: ")
    for i in hf_encoded:
        print(f'{i}: {hf_tokenizer._convert_id_to_token(i)}')


if __name__ == "__main__":
    import fire
    fire.Fire()


# vim: ts=4 sw=4 sts=4 expandtab
