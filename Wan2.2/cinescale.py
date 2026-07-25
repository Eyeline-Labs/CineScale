import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from wan.utils.utils import str2bool


ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT if (ROOT / "wan").exists() else ROOT / "Wan2.2"
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))



def parse_torch_dtype(value):
    value = value.lower()
    if value in ("fp32", "float32"):
        return torch.float32
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError(
        "Expected one of: fp32, fp16, bf16.")


def setup_distributed(args):
    rank = int(os.getenv("RANK", args.rank))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", args.device_id))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        if args.ulysses_size > 1:
            if args.ulysses_size != world_size:
                raise ValueError("--ulysses_size must equal WORLD_SIZE.")
            from wan.distributed.util import init_distributed_group

            init_distributed_group()
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError(
                "--t5_fsdp and --dit_fsdp require torchrun with multiple processes."
            )
        if args.ulysses_size > 1:
            raise ValueError(
                "--ulysses_size > 1 requires torchrun with multiple processes.")

    args.rank = rank
    args.device_id = local_rank
    return rank, world_size, local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def unique_dit_models(model):
    seen = set()
    result = []
    for name in ("low_noise_model", "high_noise_model"):
        dit_model = getattr(model, name, None)
        if dit_model is None or id(dit_model) in seen:
            continue
        seen.add(id(dit_model))
        result.append((name, dit_model))
    return result


def offload_dit_models(model):
    for _, dit_model in unique_dit_models(model):
        if next(dit_model.parameters()).device.type == "cuda":
            dit_model.to("cpu")
    torch.cuda.empty_cache()


def set_vae_dtype(vae, dtype):
    vae.dtype = dtype
    vae.mean = vae.mean.to(dtype=dtype, device=vae.device)
    vae.std = vae.std.to(dtype=dtype, device=vae.device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.model.to(device=vae.device, dtype=dtype)


def offload_vae_model(model):
    vae_model = getattr(getattr(model, "vae", None), "model", None)
    if vae_model is not None and next(vae_model.parameters()).device.type == "cuda":
        vae_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def onload_dit_models(model):
    for _, dit_model in unique_dit_models(model):
        if next(dit_model.parameters()).device.type == "cpu":
            dit_model.to(model.device)
    torch.cuda.empty_cache()


def make_model_config(wan_configs):
    cfg = copy.deepcopy(wan_configs["t2v-A14B"])
    return cfg

def parse_size(size):
    width, height = size.lower().split("*")
    return int(width), int(height)


def load_prompts(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(
            f"{path} must contain a non-empty 'prompts' list.")
    if not all(isinstance(prompt, str) and prompt.strip()
               for prompt in prompts):
        raise ValueError("Every entry in 'prompts' must be a non-empty string.")
    return prompts


def prompt_output_name(prompt, used_names):
    prefix = prompt[:30].strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._-")
    stem = stem or "prompt"
    count = used_names.get(stem, 0)
    used_names[stem] = count + 1
    if count:
        stem = f"{stem}_{count + 1:02d}"
    return f"{stem}.pt"


def best_latent_size(width, height, max_area, vae_stride, patch_size):
    aspect_ratio = height / width
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
        patch_size[1] * patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
        patch_size[2] * patch_size[2])
    return lat_h, lat_w

def _tile_starts(length, tile, stride):
    if tile >= length:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return sorted(set(values))


def _vae_decoded_frame_count(latent_frames, temporal_stride=4):
    return (latent_frames - 1) * temporal_stride + 1


def _linear_blend_1d(length, left_bound, right_bound, border_width, device,
                     dtype):
    weight = torch.ones((length,), device=device, dtype=dtype)
    border_width = int(min(border_width, length))
    if border_width <= 0:
        return weight
    ramp = (torch.arange(border_width, device=device, dtype=dtype) +
            1) / border_width
    if not left_bound:
        weight[:border_width] = ramp
    if not right_bound:
        weight[-border_width:] = torch.flip(ramp, dims=(0,))
    return weight


def _linear_blend_mask(data, is_bound, border_width):
    _, _, _, height, width = data.shape
    device = data.device
    dtype = data.dtype
    weight_h = _linear_blend_1d(height, is_bound[0], is_bound[1],
                                border_width[0], device, dtype)
    weight_w = _linear_blend_1d(width, is_bound[2], is_bound[3],
                                border_width[1], device, dtype)
    mask = torch.minimum(weight_h[:, None], weight_w[None, :])
    return mask.view(1, 1, 1, height, width)


def _vae_tile_tasks(height, width, tile_h, tile_w, stride_h, stride_w):
    tasks = []
    y_starts = _tile_starts(height, tile_h, stride_h)
    x_starts = _tile_starts(width, tile_w, stride_w)
    for y0 in y_starts:
        y1 = min(y0 + tile_h, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_w, width)
            tasks.append((y0, y1, x0, x1))
    return tasks


def _bounded_reflect_pad(size, requested):
    if requested <= 0 or size <= 1:
        return 0
    return min(requested, size - 1)


def _reflect_pad_spatial_5d(x, pad_h, pad_w):
    batch, channels, frames, height, width = x.shape
    pad_h = _bounded_reflect_pad(height, pad_h)
    pad_w = _bounded_reflect_pad(width, pad_w)
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x_4d = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height,
                                             width)
    x_4d = F.pad(x_4d, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    padded = x_4d.reshape(batch, frames, channels, height + 2 * pad_h,
                          width + 2 * pad_w).permute(0, 2, 1, 3, 4)
    return padded, (pad_h, pad_w)


def prepare_text_context(model, prompt, negative_prompt, offload_model):
    if negative_prompt == "":
        negative_prompt = model.sample_neg_prompt

    if not model.t5_cpu:
        model.text_encoder.model.to(model.device)
        context = model.text_encoder([prompt], model.device)
        context_null = model.text_encoder([negative_prompt], model.device)
        if offload_model:
            model.text_encoder.model.cpu()
    else:
        context = model.text_encoder([prompt], torch.device("cpu"))
        context_null = model.text_encoder([negative_prompt], torch.device("cpu"))
        context = [u.to(model.device) for u in context]
        context_null = [u.to(model.device) for u in context_null]

    return context, context_null


def make_unipc_scheduler(model, sample_steps, shift):
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=model.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False)
    scheduler.set_timesteps(sample_steps, device=model.device, shift=shift)
    scheduler.sigmas = scheduler.sigmas.to(model.device)
    return scheduler, scheduler.timesteps, scheduler.sigmas


def make_dpmpp_scheduler(model, sample_steps, shift):
    from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                      get_sampling_sigmas, retrieve_timesteps)

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=model.num_train_timesteps,
        solver_order=1,
        shift=1,
        use_dynamic_shifting=False)
    sampling_sigmas = get_sampling_sigmas(sample_steps, shift)
    timesteps, _ = retrieve_timesteps(
        scheduler,
        device=model.device,
        sigmas=sampling_sigmas)
    scheduler.sigmas = scheduler.sigmas.to(model.device)
    return scheduler, timesteps, scheduler.sigmas


def reset_scheduler_state(scheduler):
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    solver_order = getattr(getattr(scheduler, "config", None),
                           "solver_order", None)
    if solver_order is not None:
        if hasattr(scheduler, "model_outputs"):
            scheduler.model_outputs = [None] * solver_order
        if hasattr(scheduler, "timestep_list"):
            scheduler.timestep_list = [None] * solver_order
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "this_order"):
        scheduler.this_order = None


def move_scheduler_to_device(scheduler, device):
    if hasattr(scheduler, "sigmas") and isinstance(scheduler.sigmas,
                                                   torch.Tensor):
        scheduler.sigmas = scheduler.sigmas.to(device)
    if hasattr(scheduler, "timesteps") and isinstance(scheduler.timesteps,
                                                      torch.Tensor):
        scheduler.timesteps = scheduler.timesteps.to(device)
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [
            output.to(device) if isinstance(output, torch.Tensor) else output
            for output in scheduler.model_outputs
        ]


def set_block_tiled_self_attention(model, enabled, tile_height, tile_width,
                                   global_rope_threshold_y,
                                   global_rope_threshold_x):
    if enabled and min(tile_height, tile_width) <= 0:
        raise ValueError(
            "Block tiled self-attention tile must be positive."
        )
    if enabled and min(global_rope_threshold_y,
                       global_rope_threshold_x) < 0:
        raise ValueError(
            "Block tiled self-attention global RoPE thresholds must be "
            "non-negative."
        )
    def set_model_block_tiling(wan_model):
        target_model = getattr(wan_model, "module", wan_model)
        for block in target_model.blocks:
            block.self_attn.block_tiled_attn_enabled = enabled
            block.self_attn.block_tiled_attn_tile_h = tile_height
            block.self_attn.block_tiled_attn_tile_w = tile_width
            block.self_attn.block_tiled_attn_global_rope_threshold_y = (
                global_rope_threshold_y)
            block.self_attn.block_tiled_attn_global_rope_threshold_x = (
                global_rope_threshold_x)

    for _, dit_model in unique_dit_models(model):
        set_model_block_tiling(dit_model)


def set_spatial_ntk_rope_factor(model, factor):
    """Set Wan's spatial RoPE frequencies using an NTK extension factor."""
    if factor <= 0:
        raise ValueError("--rope_factor must be greater than zero.")

    from wan.modules.model import rope_params

    for _, dit_model in unique_dit_models(model):
        target_model = getattr(dit_model, "module", dit_model)
        head_dim = target_model.dim // target_model.num_heads
        temporal_dim = head_dim - 4 * (head_dim // 6)
        spatial_dim = 2 * (head_dim // 6)
        spatial_theta = 10000.0 * factor**(
            spatial_dim / (spatial_dim - 2))
        freq_device = target_model.patch_embedding.weight.device
        target_model.freqs = torch.cat([
            rope_params(1024, temporal_dim, theta=10000.0),
            rope_params(1024, spatial_dim, theta=spatial_theta),
            rope_params(1024, spatial_dim, theta=spatial_theta),
        ], dim=1).to(freq_device)


def predict_cond_uncond(model,
                        latent,
                        timestep,
                        context,
                        context_null,
                        seq_len,
                        guide_scale,
                        boundary,
                        offload_model,
                        y=None):
    timestep = torch.stack([timestep]).to(model.device)
    active_model = model._prepare_model_for_timestep(timestep[0], boundary,
                                                     offload_model)
    step_guide_scale = guide_scale[1] if timestep[0].item(
    ) >= boundary else guide_scale[0]

    arg_c = {"context": [context[0]], "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    if y is not None:
        arg_c["y"] = [y]
        arg_null["y"] = [y]
    latent_input = latent.to(model.device)
    latent_model_input = [latent_input]

    with torch.no_grad():
        flow_cond = active_model(latent_model_input, t=timestep, **arg_c)[0]
        flow_uncond = active_model(latent_model_input, t=timestep,
                                   **arg_null)[0]

    if offload_model:
        torch.cuda.empty_cache()

    flow = flow_uncond + step_guide_scale * (flow_cond - flow_uncond)
    return flow, flow_cond, flow_uncond

def compute_seq_len(model, latent_shape):
    _, latent_frames, lat_h, lat_w = latent_shape
    seq_len = latent_frames * lat_h * lat_w // (
        model.patch_size[1] * model.patch_size[2])
    return int(math.ceil(seq_len / model.sp_size)) * model.sp_size


def resize_latent_spatial(latent, target_h, target_w, dtype):
    if latent.shape[-2:] == (target_h, target_w):
        return latent
    z = latent.permute(1, 0, 2, 3).float()
    z = F.interpolate(
        z,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False)
    return z.permute(1, 0, 2, 3).to(dtype=dtype)


def resize_latent_to_size(model, latent, reference_size, target_size):
    reference_width, reference_height = parse_size(reference_size)
    target_width, target_height = parse_size(target_size)
    target_area = target_width * target_height
    target_lat_h, target_lat_w = best_latent_size(
        reference_width, reference_height, target_area, model.vae_stride,
        model.patch_size)
    resized = resize_latent_spatial(latent, target_lat_h, target_lat_w,
                                    model.vae.dtype)
    decoded_height = target_lat_h * model.vae_stride[1]
    decoded_width = target_lat_w * model.vae_stride[2]
    return resized, {
        "size": f"{decoded_width}*{decoded_height}",
        "latent_shape": tuple(resized.shape),
    }


def create_prompt_noise_latent(model, size, frame_num):
    width, height = parse_size(size)
    latent_h, latent_w = best_latent_size(
        width, height, width * height, model.vae_stride, model.patch_size)
    decoded_height = latent_h * model.vae_stride[1]
    decoded_width = latent_w * model.vae_stride[2]
    latent_frames = (frame_num - 1) // model.vae_stride[0] + 1
    latent = torch.randn(
        model.vae.model.z_dim,
        latent_frames,
        latent_h,
        latent_w,
        dtype=torch.float32,
        device=model.device)
    seq_len = compute_seq_len(model, latent.shape)
    metadata = {
        "size": f"{decoded_width}*{decoded_height}",
        "input_resize_mode": "none",
        "latent_shape": tuple(latent.shape),
        "input_frame_count": frame_num,
    }
    return latent, seq_len, metadata


def add_noise_to_clean_latent(clean_latent, sigma, noise):
    sigma = sigma.to(clean_latent.device).float()
    noisy = (1.0 - sigma) * clean_latent.float() + sigma * noise.float()
    return noisy.to(dtype=clean_latent.dtype).detach()


def start_index_for_step_count(round_noise_steps, sample_steps):
    if round_noise_steps < 1:
        raise ValueError("--round_noise_steps must be at least 1.")
    if round_noise_steps > sample_steps:
        raise ValueError("--round_noise_steps cannot exceed sample_steps.")
    return sample_steps - round_noise_steps


def denoise(model,
            clean_latent,
            round_noise_steps,
            context,
            context_null,
            seq_len,
            scheduler,
            timesteps,
            sigmas,
            sample_steps,
            guide_scale,
            offload_model,
            y=None):
    clean_latent = clean_latent.detach().to(model.device)
    sigmas = sigmas.to(model.device) if isinstance(sigmas,
                                                   torch.Tensor) else sigmas
    random_noise = torch.randn_like(clean_latent)
    
    start_index = start_index_for_step_count(
        round_noise_steps,
        sample_steps,
    )
    
    noisy_latent =  add_noise_to_clean_latent(
                    clean_latent.detach(),
                    sigmas[start_index],
                    random_noise
                )
    output_latent ,_ = denoise_trajectory(
        model=model,
        start_latent = noisy_latent.detach(),
        context=context,
        context_null=context_null,
        seq_len=seq_len,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=sigmas,
        sample_steps=sample_steps,
        start_index=start_index,
        guide_scale=guide_scale,
        offload_model=offload_model,
        y=y
    )

    return output_latent.detach(), start_index

def denoise_trajectory(model,
                       start_latent,
                       context,
                       context_null,
                       seq_len,
                       scheduler,
                       timesteps,
                       sigmas,
                       sample_steps,
                       start_index,
                       guide_scale,
                       offload_model,
                       step_callback=None,
                       y=None):
    boundary = model.boundary * model.num_train_timesteps
    device = model.device
    latent = start_latent.detach().to(device)
    timesteps = timesteps.to(device) if isinstance(timesteps,
                                                   torch.Tensor) else timesteps
    sigmas = sigmas.to(device) if isinstance(sigmas, torch.Tensor) else sigmas
    move_scheduler_to_device(scheduler, device)
    reset_scheduler_state(scheduler)
    move_scheduler_to_device(scheduler, device)

    for i in tqdm(
            range(start_index, sample_steps),
            desc="Denoising",
            disable=not is_main_process()):
        if step_callback is not None:
            step_callback(i)
        with torch.no_grad(), torch.amp.autocast("cuda",
                                                 dtype=model.param_dtype):
            flow, _, _ = predict_cond_uncond(
                model,
                latent,
                timesteps[i],
                context,
                context_null,
                seq_len,
                guide_scale,
                boundary,
                offload_model,
                y=y)
            flow = flow.to(device=latent.device)
            timestep = (
                timesteps[i].to(latent.device)
                if isinstance(timesteps[i], torch.Tensor) else timesteps[i])
            move_scheduler_to_device(scheduler, latent.device)
            latent = scheduler.step(
                flow.unsqueeze(0),
                timestep,
                latent.unsqueeze(0),
                return_dict=False)[0].squeeze(0).detach()

    return latent, flow

def save_video_tensor(video, save_path, fps):
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Saving videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in video.unbind(1):
            frame = ((frame.float() + 1.0) * 127.5).clamp_(0, 255)
            frame = frame.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            writer.append_data(frame)
    finally:
        writer.close()

def decode_latent_to_video_tiled(vae,
                                 latent,
                                 tile_h=None,
                                 tile_w=None,
                                 stride_h=160,
                                 stride_w=140):
    
    vae_model = vae.model
    device = vae.device
    reflect_pad = 24

    z = latent.to(device).unsqueeze(0)
    _, _, latent_frames, latent_h, latent_w = z.shape
    tile_h = latent_h if tile_h is None else min(tile_h, latent_h)
    stride_h = tile_h if stride_h is None else min(stride_h, tile_h)
    tile_w = latent_w if tile_w is None else min(tile_w, latent_w)
    stride_w = tile_w if stride_w is None else min(stride_w, tile_w)
    blend_h = tile_h - stride_h
    blend_w = tile_w - stride_w
    out_frames = _vae_decoded_frame_count(latent_frames)
    out_h, out_w = latent_h * 8, latent_w * 8


    with torch.no_grad(), torch.amp.autocast("cuda", dtype=vae.dtype):

        z, (pad_h, pad_w) = _reflect_pad_spatial_5d(z, reflect_pad, reflect_pad)
        tasks = _vae_tile_tasks(latent_h, latent_w, tile_h, tile_w, stride_h, stride_w)

        values = torch.zeros(1, 3, out_frames, out_h, out_w, dtype=torch.float32)
        weights = torch.zeros(1, 1, out_frames, out_h, out_w, dtype=torch.float32)

        for y0, y1, x0, x1 in tqdm(tasks, desc="VAE Decode", unit="tile",
                                    disable=not is_main_process()):
            tile = z[:, :, :, y0:y1 + 2 * pad_h, x0:x1 + 2 * pad_w].to(device)
            tile_out = vae_model.decode(tile, vae.scale).float().cpu()
            core_h, core_w = (y1 - y0) * 8, (x1 - x0) * 8
            tile_out = tile_out[..., pad_h * 8:pad_h * 8 + core_h,
                                     pad_w * 8:pad_w * 8 + core_w]

            out_y0, out_x0 = y0 * 8, x0 * 8
            mask = _linear_blend_mask(
                tile_out,
                is_bound=(y0 == 0, y1 >= latent_h, x0 == 0, x1 >= latent_w),
                border_width=(blend_h * 8, blend_w * 8)).float().cpu()
            values[:, :, :, out_y0:out_y0 + core_h, out_x0:out_x0 + core_w] += tile_out * mask
            weights[:, :, :, out_y0:out_y0 + core_h, out_x0:out_x0 + core_w] += mask

            del tile, tile_out, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    vae_model.clear_cache()
    return (values / weights.clamp_min(1e-6)).clamp_(-1, 1).squeeze(0).contiguous()


def save_latent_video_tiled(vae,
                            latent,
                            save_path,
                            fps,
                            tile_h=None,
                            tile_w=128):
    
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Saving videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc
    
    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)
    video = decode_latent_to_video_tiled(
        vae,
        latent,
        tile_h=tile_h,
        tile_w=tile_w)
    for frame in video.unbind(1):
        frame = ((frame.float() + 1.0) * 127.5).clamp_(0, 255)
        frame = frame.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        writer.append_data(frame)
    writer.close()        

def load_latent_payload(path, latent_key="final_latent"):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        if latent_key not in ("tensor", "final_latent"):
            raise KeyError(
                f"{path} is a raw tensor and cannot provide latent key "
                f"'{latent_key}'.")
        return payload, {}
    metadata = dict(payload.get("metadata", {}))

    if latent_key in payload:
        latent = payload[latent_key]
        if latent is None:
            raise KeyError(f"{path} contains '{latent_key}', but it is None.")
        metadata["decoded_latent_key"] = latent_key
        return latent, metadata

    if latent_key != "final_latent":
        available = sorted(k for k in payload.keys() if k != "metadata")
        raise KeyError(
            f"{path} does not contain '{latent_key}'. Available latent keys: {available}")
    if "final_latent" in payload:
        metadata["decoded_latent_key"] = "final_latent"
        return payload["final_latent"], metadata
    if "clean_latent" in payload:
        metadata["decoded_latent_key"] = "clean_latent"
        return payload["clean_latent"], metadata
    raise KeyError(
        f"{path} must contain a tensor, 'final_latent', or 'clean_latent'.")


def decode_latent_only(args, cfg):
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", args.device_id))
    if rank != 0:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from wan.modules.vae2_1 import Wan2_1_VAE

    latent_path = Path(args.decode_latent)
    output_path = latent_path.with_suffix(".mp4")
    latent, metadata = load_latent_payload(
        latent_path, args.decode_latent_key)
    fps = metadata.get("fps", args.fps)
    vae_dtype = parse_torch_dtype(args.vae_dtype)
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        dtype=vae_dtype,
        device=torch.device(f"cuda:{local_rank}"))
    set_vae_dtype(vae, vae_dtype)
    if metadata.get("decoded_latent_key") == "prompt_base_latent":
        with torch.no_grad():
            video = vae.decode([latent.to(vae.device)])[0]
        save_video_tensor(video, output_path, fps)
    else:
        save_latent_video_tiled(vae, latent, output_path, fps)

    print(f"Saved video as {output_path}", flush=True)


def run(args, model, cfg, prompt, output_latent, prompt_index):

    frame_num = args.frame_num or cfg.frame_num
    sample_steps = args.sample_steps or cfg.sample_steps
    sample_shift = args.sample_shift or cfg.sample_shift
    guide_scale = args.sample_guide_scale or cfg.sample_guide_scale
    guide_scale = (guide_scale, guide_scale) if isinstance(
        guide_scale, float) else guide_scale
    step_offload_model = args.offload_model and not args.dit_fsdp
    text_offload_model = args.offload_model and not args.t5_fsdp

    if frame_num % 4 != 1:
        raise ValueError("--frame_num must be 4n+1 for Wan T2V.")
   
    encode_size = args.size
    global_rope_threshold_y = (
        args.block_tiled_self_attn_global_rope_threshold_vertical
        if args.block_tiled_self_attn_global_rope_threshold_vertical
        is not None else args.block_tiled_self_attn_global_rope_threshold)
    global_rope_threshold_x = (
        args.block_tiled_self_attn_global_rope_threshold_horizontal
        if args.block_tiled_self_attn_global_rope_threshold_horizontal
        is not None else args.block_tiled_self_attn_global_rope_threshold)

    context = context_null = None
    restart_scheduler = restart_timesteps = restart_sigmas = None
   
    context, context_null = prepare_text_context(
        model, prompt, args.negative_prompt, text_offload_model)
    restart_scheduler, restart_timesteps, restart_sigmas = make_dpmpp_scheduler(
        model, sample_steps, sample_shift)
    if args.offload_model and (args.t5_fsdp or args.dit_fsdp):
        onload_dit_models(model)

    # Every prompt base is generated at Wan's native resolution with the
    # original RoPE frequencies. This also prevents an NTK factor from a
    # previous prompt's refinement pass leaking into the next prompt.
    set_spatial_ntk_rope_factor(model, 1.0)

    def configure_block_tiled_attention(enabled):
        set_block_tiled_self_attention(
            model,
            enabled,
            args.block_tiled_self_attn_tile_height,
            args.block_tiled_self_attn_tile_width,
            global_rope_threshold_y,
            global_rope_threshold_x)

    prompt_base_size = "1280*720"
    base_latent, base_seq_len, _ = create_prompt_noise_latent(
        model, prompt_base_size, frame_num)
    if is_main_process():
        logging.info("Generating prompt-conditioned base latent at %s",
                     prompt_base_size)
    base_scheduler, base_timesteps, base_sigmas = make_unipc_scheduler(
        model, sample_steps, sample_shift)
    set_block_tiled_self_attention(
        model,
        False,
        args.block_tiled_self_attn_tile_height,
        args.block_tiled_self_attn_tile_width,
        global_rope_threshold_y,
        global_rope_threshold_x)
    base_latent, _ = denoise_trajectory(
        model=model,
        start_latent=base_latent,
        context=context,
        context_null=context_null,
        seq_len=base_seq_len,
        scheduler=base_scheduler,
        timesteps=base_timesteps,
        sigmas=base_sigmas,
        sample_steps=sample_steps,
        start_index=0,
        guide_scale=guide_scale,
        offload_model=step_offload_model,
        y=None)
    del base_scheduler, base_timesteps, base_sigmas
    prompt_base_latent_cpu = (
        base_latent.detach().cpu() if is_main_process() else None)

    clean_latent, resize_metadata = resize_latent_to_size(
        model, base_latent, prompt_base_size, encode_size)
    seq_len = compute_seq_len(model, clean_latent.shape)
    metadata = {
        **resize_metadata,
        "input_mode": "prompt",
        "prompt_base_size": prompt_base_size,
        "prompt_base_latent_shape": tuple(base_latent.shape),
        "input_resize_mode": "latent",
        "input_frame_count": frame_num,
        "block_tiled_self_attn": args.block_tiled_self_attn,
        "rope_factor": (
            1.0 if args.block_tiled_self_attn else args.rope_factor),
    }
    del base_latent
    if is_main_process():
        logging.info("Generated prompt base at %s, then latent-resized to %s",
                     metadata["prompt_base_size"], metadata["size"])

    if args.offload_model:
        offload_vae_model(model)

    configure_block_tiled_attention(args.block_tiled_self_attn)
    if not args.block_tiled_self_attn:
        set_spatial_ntk_rope_factor(model, args.rope_factor)
        if is_main_process():
            logging.info(
                "Using spatial NTK-RoPE factor %.4g for high-resolution "
                "refinement", args.rope_factor)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(model.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = (
        noop_no_sync if model.high_noise_model is model.low_noise_model else
        getattr(model.high_noise_model, "no_sync", noop_no_sync))
    final_latent = clean_latent.detach()
    with no_sync_low(), no_sync_high():
        final_latent, start_index = denoise(
            model=model,
            clean_latent=final_latent,
            round_noise_steps=args.round_noise_steps,
            context=context,
            context_null=context_null,
            seq_len=seq_len,
            scheduler=restart_scheduler,
            timesteps=restart_timesteps,
            sigmas=restart_sigmas,
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            offload_model=step_offload_model,
            y=None)

    if args.offload_model:  
        offload_dit_models(model)

    if is_main_process():
        payload = {
            "final_latent": final_latent.detach().cpu(),
            "prompt_base_latent": prompt_base_latent_cpu,
            "metadata": {
                **metadata,
                "fps": args.fps,
                "prompt": prompt,
                "prompt_index": prompt_index,
                "run_seed": args.run_seed,
            },
        }
        output_latent.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_latent)
        print(f"Saved video as {output_latent}", flush=True)

    del clean_latent, context, context_null
    del restart_scheduler, restart_timesteps, restart_sigmas
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 CineScale for every prompt in a JSON file."
    )
    parser.add_argument(
        "--prompts_json",
        default=str(ROOT / "prompts.json"),
        help="JSON file containing a top-level 'prompts' list.")
    parser.add_argument(
        "--output_dir",
        default="CineScale/batch_latents",
        help="Directory for prompt-named latent .pt outputs.")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Wan2.2-T2V-A14B checkpoint directory.")
    parser.add_argument(
        "--decode_latent",
        default=None,
        help="Decode this saved latent .pt to a same-named .mp4 and exit.")
    parser.add_argument(
        "--decode_latent_key",
        default="final_latent",
        help="Tensor to decode from the saved .pt, such as final_latent or prompt_base_latent.")
    parser.add_argument(
        "--vae_dtype",
        default="fp16",
        choices=("fp32", "fp16", "bf16"),
        help="VAE encode/decode dtype.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--size",
        default="3840*2160",
        help="Target area as width*height. Prompt mode starts from a 1280*720 latent; latent input mode preserves the input latent aspect ratio.")
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--round_noise_steps",
        type=int,
        default=30,
        help="Exact denoising steps per noise round.")
    parser.add_argument("--sample_shift", type=float, default=12.0)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--block_tiled_self_attn",
        type=str2bool,
        default=True,
        help="Tile only DiT self-attention inside each block, stitch the self-attention output, then run global cross-attention/FFN.")
    parser.add_argument(
        "--rope_factor",
        type=float,
        default=1.0,
        help="Spatial NTK-RoPE extension factor used during high-resolution "
        "refinement. Only valid when --block_tiled_self_attn=false.")
    parser.add_argument(
        "--block_tiled_self_attn_tile_width",
        type=int,
        default=32,
        help="Inner self-attention tile width in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_tile_height",
        type=int,
        default=18,
        help="Inner self-attention tile height in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_global_rope_threshold",
        type=float,
        default=20.0,
        help="Fallback threshold for both axes. Axis-specific options override it.")
    parser.add_argument(
        "--block_tiled_self_attn_global_rope_threshold_horizontal",
        type=float,
        default=16,
        help="Horizontal uncompressed distance threshold in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_global_rope_threshold_vertical",
        type=float,
        default=9,
        help="Vertical uncompressed distance threshold in transformer patch-token units.")
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Sequence-parallel world size for DiT. With torchrun, set this to nproc_per_node."
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Shard the T5 text encoder with FSDP in distributed runs.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Shard the Wan DiT models with FSDP in distributed runs.")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import wan
    from wan.configs import WAN_CONFIGS

    cfg = make_model_config(WAN_CONFIGS)
    args.model_version = "2.2"
    logging.info("Using Wan2.2 checkpoint layout")
    if args.decode_latent is not None:
        decode_latent_only(args, cfg)
        return
    prompts = load_prompts(args.prompts_json)

    rank, world_size, _ = setup_distributed(args)

    if args.offload_model is None:
        args.offload_model = True
        if is_main_process():
            logging.info("offload_model not specified; using %s",
                         args.offload_model)

    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError(
            f"cfg.num_heads={cfg.num_heads} must be divisible by --ulysses_size."
        )
    if args.rope_factor <= 0:
        raise ValueError("--rope_factor must be greater than zero.")
    if args.block_tiled_self_attn and args.rope_factor != 1.0:
        raise ValueError(
            "--rope_factor is only valid when "
            "--block_tiled_self_attn=false.")

    seed = args.base_seed if args.base_seed >= 0 else (
        random.randint(0, sys.maxsize) if rank == 0 else 0)
    if dist.is_initialized():
        seed_holder = [seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_holder, src=0)
        seed = seed_holder[0]
    random.seed(seed)
    torch.manual_seed(seed)
    args.run_seed = seed


    model = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=args.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype)
    model.model_version = "2.2"

    set_vae_dtype(model.vae, parse_torch_dtype(args.vae_dtype))
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Loaded %d prompts from %s", len(prompts),
                     args.prompts_json)
    if dist.is_initialized():
        dist.barrier()

    used_names = {}
    for prompt_index, prompt in enumerate(prompts):
        output_latent = output_dir / prompt_output_name(prompt, used_names)
        if is_main_process():
            logging.info("Running prompt %d/%d: %s", prompt_index + 1,
                         len(prompts), prompt[:80])
        run(
            args,
            model,
            cfg,
            prompt,
            output_latent,
            prompt_index)
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

