<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT-Chinese

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) 

</div>


<!-- # ⚡ Lit-GPT-Chinese -->

Hackable [implementation](lit_gpt/model.py) of state-of-the-art open-source large language models for **chinese** released under the **Apache 2.0 license**.

Supports the following popular model checkpoints (along with all the english models supported by official Lit-GPT):

| Model and usage                                                                   | Model size                               | Reference                                                                                        |
|-----------------------------------------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------|
| Baichuan 2                                | 7B-Chat/Base, 13B-Chat/Base | [Baichuan 2](https://github.com/baichuan-inc/Baichuan2)                                         |
| ChatGLM3                                 | 6B, 6B-Base, 6B-32k | [ChatGLM3](https://github.com/THUDM/ChatGLM3)                                         |
| ChatGLM2                                 | 6B | [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)                                         |


This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

&nbsp;

## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

&nbsp;

## Setup

Clone the repo:

```bash
git clone https://github.com/metame-none/lit-gpt-chinese
cd lit-gpt-chinese
```

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Install with all dependencies (including quantization, sentencepiece, tokenizers for Llama models, etc.):

```bash
pip install -r requirements-all.txt
```

**(Optional) Use Flash Attention 2**

Flash Attention 2 will be used automatically if PyTorch 2.2 (or higher) is installed.
Currently, that requires installing PyTorch nightly, which you can get by running:

```bash
pip uninstall -y torch torchvision torchaudio torchtext
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

You are all set! 🎉

&nbsp;

## Use the model

**Take ChatGLM3-6B as an example:**

1. Download repo and checkpoints (manually or using `git lfs`):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm3-6b $path
```
2. Convert the checkpoint to the Lit-GPT format:
```bash
ln -snf $path checkpoints/chatglm/chatglm3-6b-hf

python scripts/convert_hf_checkpoint.py --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf
```
3. Iteratively generate responses:
```bash
python chat/base.py --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf  --precision "16-true"
```

<details>
<summary> Optional: check the lit-gpt model is numerically equivalent to the original model. </summary>

- make the following changes to the original model (modeling_chatglm.py):

```diff
-from .configuration_chatglm import ChatGLMConfig
+from configuration_chatglm import ChatGLMConfig

 # flags required to enable jit fusion kernels

@@ -157,7 +157,7 @@ class RotaryEmbedding(nn.Module):
         )


-@torch.jit.script
+# @torch.jit.script
 def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
```
- check the model difference:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tests/test_chatglm3.py model_diff ./checkpoints/chatglm/chatglm3-6b-hf
```

- for baichuan2 model:
```diff
--- a/modeling_baichuan.py
+++ b/modeling_baichuan.py
@@ -20,8 +20,8 @@
 # limitations under the License.


-from .configuration_baichuan import BaichuanConfig
-from .generation_utils import build_chat_input, TextIterStreamer
+from configuration_baichuan import BaichuanConfig
+from generation_utils import build_chat_input, TextIterStreamer

 import math
 from typing import List, Optional, Tuple, Union
```

</details>

&nbsp;

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the random 10k samples from [multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_belle_chatglm3.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py --data_dir ./data/belle_chat_ramdon_10k_chatglm3 --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf --out_dir out/adapter/belle_chatglm3_6b --precision "bf16-true"

# test the finetuned model
python chat/adapter.py --adapter_path ./out/adapter/belle_chatglm3_6b/lit_model_adapter_finetuned.pth --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf --precision "16-true"
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py --data_dir ./data/belle_chat_ramdon_10k_chatglm3 --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf --out_dir out/adapter_v2/belle_chatglm3_6b --precision "bf16-true"

# test the finetuned model
python chat/adapter_v2.py --adapter_path ./out/adapter_v2/belle_chatglm3_6b/lit_model_adapter_finetuned.pth --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf --precision "16-true"
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py --data_dir ./data/belle_chat_ramdon_10k_chatglm3 --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf --out_dir out/lora/belle_chatglm3_6b --precision "16-true"

# test the finetuned model
python chat/lora.py --lora_path ./out/lora/belle_chatglm3_6b/lit_model_lora_finetuned.pth --checkpoint_dir ./checkpoints/chatglm/chatglm3-6b-hf  --precision "16-true"
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

## Reference

For more details, please refer to the [Lit-GPT](https://github.com/Lightning-AI/lit-gpt)

## License

Lit-GPT-Chinese is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
