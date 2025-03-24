# DeepSick-R1

### *"Too much feeling sick while reproducing DeepSeek-R1!!"*

#### 🚑 Why do we need to see this repository although there are many open-source codes for building DeepSeek-R1?

- My short code lines and a few code files make users happy.
- This code doesn't use huggingface GRPOTrainer class which may bring in frustration because of too much complexities when users customize GRPOTrainer to fit individual research and production.
- This code has only three files (main.py, trainer.py, and utils.py) to know for training, while famous repositories [Open-R1](https://github.com/huggingface/open-r1), [R1-V](https://github.com/Deep-Agent/R1-V), [verl](https://github.com/volcengine/verl), and [TinyZero](https://github.com/Jiayi-Pan/TinyZero) have 1000+ code files, many config files, and too much folders.
- [vLLM](https://github.com/vllm-project/vllm) is applied so that users can generate answer candidates realy fastly.
- Although [vLLM](https://github.com/vllm-project/vllm) is applied, total number of code lines is still short.
- For training with multiple GPU, one GPU will be assigned to vLLM model to generate, and the other GPUs are focusing on training.

**Requirements!!: This repository requires two GPUs at least, because vLLM should be assigned to another GPU in order to separate the training GPU and inference GPU.**

---

## 🚀 Short Talks

- When we train Qwen2-VL-3B-Instruct with 100k QA samples on 2 NVIDIA A100 80GB VRAM, it takes 14 hours to train.
- The GPU memory usage was 40~60GB when unfreezing all MLP parameters in LLM decoder part, where I use 2 batch, 4 number of generations, and 4 GRPO iterations. 
- This repository is dealing with vision language models (VLMs) only, but I believe this code is really easy, so users can easily modify the code for LLM version.
- In the current version, Qwen2.5-VL and latest vLLM are not supported because there is first flash attention issue in latest vLLM version and model parameter access issues. I will let this code updated once it is all resolved.

---

### 🍉 Install

```bash
#!/bin/bash
conda create -n deepsick python=3.12 -y
conda activate deepsick

# install vllm [Error happens using FlashAttention when using latest vllm]
pip install vllm==0.7.2

# install package
pip install trl wandb debugpy datasets deepspeed accelerate

# flash attention
pip install flash-attn --no-build-isolation
```

---

### 🍲 What to see for understanding

```shell
# Total 825 lines
main.py (286 lines)
trainer.py (108 lines)
utils.py (431 lines)
```

---

### 💻 Training with multi-GPU 

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)-ZeRO3 is used.
```shell
# ds_accel.yaml is the config file for deepspeed zero3
bash train.sh
```

