## ___***CineScale: Free Lunch in High-Resolution Cinematic Visual Generation***___

### CineScale is an extended work of [FreeScale](https://github.com/ali-vilab/FreeScale) for higher-resolution visual generation, unlocking the 4k video generation!

<div align="center">
 <a href='https://arxiv.org/abs/2508.15774'><img src='https://img.shields.io/badge/arXiv-2508.15774-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://eyeline-labs.github.io/CineScale/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

_**[Haonan Qiu](http://haonanqiu.com/), [Ning Yu*](https://ningyu1991.github.io/), [Ziqi Huang](https://ziqihuangg.github.io/), [Paul Debevec](https://www.pauldebevec.com/), and [Ziwei Liu*](https://liuziwei7.github.io/)**_
<br>
(* Corresponding Author)

From Nanyang Technological University and Netflix Eyeline Studios.

[![Watch the video](https://img.youtube.com/vi/bDYmXpNctc4/0.jpg)](https://www.youtube.com/watch?v=bDYmXpNctc4).     
(Click to enjoy 4k demo 📹)
</div>


## ⚙️ Setup

### Install Environment via Anaconda
```bash
git clone https://github.com/Eyeline-Labs/CineScale.git
cd CineScale

conda create -n cinescale python=3.10
conda activate cinescale
pip install -e .
pip install xfuser>=0.4.3
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 🧰 Models

|Model|Tuning Resolution|Checkpoint|Description
|:---------|:---------|:--------|:--------|
|CineScale-1.3B-T2V (Text2Video)|1088x1920|[Hugging Face](https://huggingface.co/Eyeline-Labs/CineScale/blob/main/t2v_1.3b_ntk20.ckpt)|Support 3k(1632x2880) inference on A100 x 1
|CineScale-14B-T2V (Text2Video)|1088x1920|[Hugging Face](https://huggingface.co/Eyeline-Labs/CineScale/blob/main/t2v_14b_ntk20.ckpt)|Support 4k(2176x3840) inference on A100 x 8
|CineScale-14B-I2V (Text2Video)|1088x1920|[Hugging Face](https://huggingface.co/Eyeline-Labs/CineScale/blob/main/i2v_14b_ntk20.ckpt)|Support 4k(2176x3840) inference on A100 x 8

## 💫 Inference with Command
### Model Preparation

Download the checkpoint from [Hugging Face](https://huggingface.co/Eyeline-Labs/CineScale/tree/main) and put it the folder `models`.

### 2K-Resolution Text-to-Video (Base Model Wan2.1-1.3B)

Single GPU
```bash
  CUDA_VISIBLE_DEVICES=0 python cinescale_t2v1.3b_single.py
```
Multiple GPUs
```bash
  torchrun --standalone --nproc_per_node=8 cinescale_t2v1.3b.py
```

### 3K-Resolution Text-to-Video (Base Model Wan2.1-1.3B)

```bash
  torchrun --standalone --nproc_per_node=8 cinescale_t2v1.3b_pro.py
```

### 4K-Resolution Text-to-Video (Base Model Wan2.1-14B)

```bash
  torchrun --standalone --nproc_per_node=8 cinescale_t2v14b_pro.py
```

### 4K-Resolution Image-to-Video (Base Model Wan2.1-14B)

```bash
  torchrun --standalone --nproc_per_node=8 cinescale_i2v14b.py
```


## 🤗 Acknowledgements
This codebase is built on top of the open-source implementation of [Wan2.1](https://github.com/Wan-Video/Wan2.1) based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo) repo.

## 😉 Citation
```bib
@article{qiu2025cinescale,
  title={CineScale: Free Lunch in High-Resolution Cinematic Visual Generation}, 
  author={Haonan Qiu and Ning Yu and Ziqi Huang and Paul Debevec and Ziwei Liu},
  journal={arXiv preprint arXiv:2508.15774},
  year={2025}
}
```