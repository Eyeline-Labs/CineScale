[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2508.15774)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://eyeline-labs.github.io/CineScale/)

> **Note:** This repository is under construction. 


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Figures/CineScale_Header_Dark.png">
    <source media="(prefers-color-scheme: light)" srcset="Figures/CineScale_Header_Light.png">
    <img src="Figures/CineScale_Header_Light.png" alt="CineScale: Open-Sourcing Tuning-Free High-Resolution Video Generation" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://gordonchen19.github.io">Gordon Chen</a><sup>†</sup>,
  <a href="http://haonanqiu.com/">Haonan Qiu</a><sup>†</sup>,
  <a href="https://ningyu1991.github.io/">Ning Yu</a><sup>*</sup>,
  <a href="https://ziqihuangg.github.io">Ziqi Huang</a>,
  <a href="https://www.pauldebevec.com/">Paul Debevec</a>,
  <a href="https://liuziwei7.github.io/">Ziwei Liu</a><sup>*</sup>
</p>

<p align="center">
  <sup>†</sup> Equal contribution &nbsp;&nbsp; <sup>*</sup> Corresponding authors
</p>


<p align="center">From Nanyang Technological University and Netflix Eyeline Studios.</p>


## ⚡ TL;DR

Most video generators are trained at limited spatial resolutions due to the scarcity of high-resolution 4K video data and the prohibitive computational cost of large-scale training on such data. Most video diffusion models are trained on 720p videos and are therefore effectively limited to generating videos at similar resolutions during inference. To address this gap, we propose CineScale. CineScale, to the best of our knowledge, is the first tuning-free inference framework enabling pretrained video diffusion models to generate high-fidelity videos at resolutions far beyond those encountered during training, without any fine-tuning.

CineScale unlocks tuning-free 4K video generation ([Watch our Video Demo here](https://eyeline-labs.github.io/CineScale/))!

## 🎬 Qualitative Results

<p align="center">
  <img src="Figures/Teaser_2.png" alt="Qualitative 4K video generation results from CineScale" width="100%">
</p>

<p align="center"><em>Qualitative 4K video generation results produced by CineScale.</em></p>

<p align="center">
  <img src="Figures/4K_Comparison.png" alt="Qualitative comparison of CineScale with existing video generation models" width="100%">
</p>

<p align="center"><em>Qualitative comparison with existing video generation models.</em></p>


## 📊 Quantitative Results

CineScale substantially improves perceptual quality at high resolution, with
the strongest gains in aesthetic and imaging quality. Subject consistency,
background consistency, and motion smoothness remain comparable to the base
model because CineScale focuses on recovering fine-grained spatial details
while preserving the temporal behavior and semantic structure established by
the low-resolution generation.

**VBench comparison across different target resolutions.** SC: Subject Consistency; BC: Background Consistency; TF: Temporal Flickering; AQ: Aesthetic Quality; IQ: Imaging Quality. Higher is better for every metric. The best, second-best, and third-best results are marked in **bold**, <u>underline</u>, and *italics*, respectively.

| Method | SC ↑ | BC ↑ | TF ↑ | AQ ↑ | IQ ↑ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| *Tuning-Free* | | | | | | |
| Wan2.1-720p | 0.9570 | 0.9605 | 0.9845 | 0.5646 | 0.6828 | 0.8299 |
| Wan2.1-1K | 0.9540 | 0.9645 | 0.9898 | 0.4989 | 0.5826 | 0.7980 |
| Wan2.1-4K | 0.9470 | 0.9760 | <u>0.9950</u> | 0.2880 | 0.3740 | 0.7160 |
| CineScale-2K (Ours) | *0.9734* | <u>0.9777</u> | 0.9795 | **0.6488** | <u>0.7156</u> | **0.8581** |
| *Tuning-Based* | | | | | | |
| UltraWan-1K | 0.9586 | 0.9661 | 0.9853 | 0.5686 | 0.6966 | 0.8350 |
| UltraWan-4K | 0.9581 | 0.9611 | 0.9771 | 0.5769 | *0.7144* | 0.8375 |
| UltraGen-1080P | <u>0.9771</u> | <u>0.9777</u> | **0.9961** | 0.5819 | **0.7350** | <u>0.8536</u> |
| UltraGen-4K | **0.9854** | **0.9894** | *0.9933* | 0.5787 | 0.6832 | *0.8460* |
| LUVE-2K | 0.9583 | 0.9676 | 0.9818 | <u>0.5978</u> | 0.7115 | 0.8434 |
| LUVE-4K | 0.9536 | 0.9646 | 0.9809 | *0.5891* | 0.7133 | 0.8403 |


## ⚙️ Setup

The setup follows exactly from the original Wan2.2 Repository. (https://github.com/Wan-Video/Wan2.2)


## 💫 Usage

Deffine your prompts in:

```bash
Wan2.2/prompts.json
```

For instance:

```bash
{
    "prompts": [
        "A little girl, lost in the city and separated from her parents in New York's Times Square, looks up. **The camera tilts up**, following her gaze. Starting from the ground, it slowly reveals the massive, glittering, and dizzying skyscrapers and billboards, powerfully emphasizing her smallness and helplessness in a vast world."
    ]   
}
```

and then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=5 CineScale/Wan2.2/cinescale.py \
  --prompts_json CineScale/Wan2.2/prompts_2.json \
  --output_dir CineScale/example_videos \
  --ckpt_dir Wan2.2-T2V-A14B \
  --frame_num 41 \
  --round_noise_steps 25 \
  --ulysses_size 5 \
  --dit_fsdp \
  --t5_cpu \
  --offload_model true 
```

### Video-to-Video (V2V)

Add `--input_video` to refine an existing video. The prompt in `prompts.json`
guides the refinement.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=5 CineScale/Wan2.2/cinescale.py \
  --input_video path/to/input.mp4 \
  --prompts_json CineScale/Wan2.2/prompts.json \
  --output_dir CineScale/v2v_latents \
  --ckpt_dir Wan2.2-T2V-A14B \
  --size '3840*2160' \
  --frame_num 41 \
  --round_noise_steps 25 \
  --ulysses_size 5 \
  --dit_fsdp \
  --t5_cpu \
  --offload_model true
```

CineScale samples up to `--frame_num` frames, VAE-encodes the video, bilinearly
upsamples its latent to `--size`, and runs high-resolution refinement. Keep
`--round_noise_steps` below `--sample_steps`; a full restart discards the input.

Wan2.2 TI2V-5B checkpoints are also supported. The model
variant is detected from the standard checkpoint layout, or it can be selected
explicitly:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python CineScale/Wan2.2/cinescale.py \
  --prompts_json CineScale/Wan2.2/prompts.json \
  --output_dir CineScale/example_videos_5b \
  --ckpt_dir Wan2.2-TI2V-5B \
  --model_variant ti2v-5B \
  --offload_model true \
  --convert_model_dtype \
  --t5_cpu
```

The 5B path uses its native `1280*704` prompt base, Wan2.2 VAE, 24 FPS, 50
sampling steps, and shift 5.0 unless those settings are overridden.

Wan2.1 T2V-1.3B checkpoints are supported through the single-DiT path. Its
standard checkpoint layout can be detected automatically, or selected
explicitly. It generates its prompt base at `832*480` while using the same
80-by-45 maximum-reference grid as Wan2.2 A14B, with 21-by-12 query tiles,
horizontal/vertical thresholds of 16/9, and maximum zero-based relative
offsets `x=79` and `y=44`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 CineScale/Wan2.2/cinescale.py \
  --prompts_json CineScale/Wan2.2/prompts.json \
  --output_dir CineScale/example_videos_1_3b \
  --ckpt_dir Wan2.1-T2V-1.3B \
  --model_variant wan2.1-t2v-1.3B \
  --block_tiled_self_attn true \
  --frame_num 41 \
  --round_noise_steps 40 \
  --ulysses_size 4 \
  --dit_fsdp \
  --t5_cpu \
  --offload_model true
```

This path uses an `832*480` prompt base, the Wan2.1 VAE, a single DiT,
16 FPS, 50 sampling steps, and shift 8.0 unless overridden.
When decoding its high-resolution latent payloads, CineScale automatically uses
`64*64` VAE latent tiles in both spatial dimensions. The tile size can be
overridden with `--vae_decode_tile_height` and `--vae_decode_tile_width`.

To decode the video, run: 

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python CineScale/Wan2.2/cinescale.py \
  --decode_latent path/to/video_latent.pt \
  --ckpt_dir checkpoint/to/model_weights/Wan2.2-T2V-A14B
```

To inspect, batch-decode, and optionally score every completed latent payload,
use `vbench_batch.py`. The VAE is loaded once and the decoded MP4 files are
written to a separate directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
python CineScale/Wan2.2/vbench_batch.py \
  --latents_dir CineScale/example_videos_1_3b \
  --videos_dir CineScale/vbench_videos_1_3b \
  --prompts_json CineScale/Wan2.2/prompts.json \
  --ckpt_dir Wan2.1-T2V-1.3B \
  --model_variant wan2.1-t2v-1.3B \
  --latent_key final_latent \
  --run_vbench
```

The script prints the stored prompt for every `.pt`, skips existing MP4 files
unless `--overwrite` is supplied, writes a JSON manifest beside the video
directory, and runs the six dimensions supported by VBench custom-input mode.
Use `--list_only` to inspect payloads without decoding them.



## 🤗 Acknowledgements
This codebase is built on top of the open-source implementation of [Wan2.2](https://github.com/Wan-Video/Wan2.2) repository.

## 📖 Citation
If you find CineScale useful in your research or projects, consider citing our paper:
```bib
@article{chen2026cinescale,
  title={CineScale: Tuning-Free High-Resolution Video Generation},
  author={Chen, Gordon and Qiu, Haonan and Yu, Ning and Huang, Ziqi and Debevec, Paul and Liu, Ziwei},
  journal={arXiv preprint arXiv:2508.15774},
  year={2026}
}
```
