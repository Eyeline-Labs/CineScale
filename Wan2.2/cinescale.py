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
from contextlib import ExitStack, contextmanager
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


MODEL_VARIANTS = {
    "t2v-A14B": {
        "pipeline": "WanT2V",
        "prompt_base_size": "1280*720",
        "vae_version": "2.1",
        "model_version": "2.2",
        "ar_max_relative_y": 44,
        "ar_max_relative_x": 79,
        "vae_decode_tile_height": None,
        "vae_decode_tile_width": 128,
    },
    "ti2v-5B": {
        "pipeline": "WanTI2V",
        "prompt_base_size": "1280*704",
        "vae_version": "2.2",
        "model_version": "2.2",
        "ar_max_relative_y": 44,
        "ar_max_relative_x": 79,
        "vae_decode_tile_height": None,
        "vae_decode_tile_width": 128,
    },
    "wan2.1-t2v-1.3B": {
        "pipeline": "WanT2VSingle",
        "prompt_base_size": "832*480",
        "vae_version": "2.1",
        "model_version": "2.1",
        "ar_max_relative_y": 44,
        "ar_max_relative_x": 79,
        "vae_decode_tile_height": 64,
        "vae_decode_tile_width": 64,
    },
}


def parse_model_variant(value):
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "14b": "t2v-A14B",
        "a14b": "t2v-A14B",
        "t2v-a14b": "t2v-A14B",
        "5b": "ti2v-5B",
        "ti2v-5b": "ti2v-5B",
        "1.3b": "wan2.1-t2v-1.3B",
        "t2v-1.3b": "wan2.1-t2v-1.3B",
        "wan2.1-1.3b": "wan2.1-t2v-1.3B",
        "wan2.1-t2v-1.3b": "wan2.1-t2v-1.3B",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = "auto, t2v-A14B, ti2v-5B, wan2.1-t2v-1.3B"
        raise argparse.ArgumentTypeError(
            f"Unknown model variant '{value}'. Expected one of: {choices}.") from exc


def _checkpoint_config(checkpoint_dir):
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def detect_model_variant(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_config = _checkpoint_config(checkpoint_dir)
    model_type = str(checkpoint_config.get("model_type", "")).lower()
    directory_name = checkpoint_dir.name.lower()

    if ((checkpoint_dir / "low_noise_model").is_dir()
            and (checkpoint_dir / "high_noise_model").is_dir()):
        return "t2v-A14B"
    if (checkpoint_dir / "Wan2.2_VAE.pth").is_file():
        return "ti2v-5B"
    if (("1.3b" in directory_name and "t2v" in directory_name)
            or (model_type == "t2v"
                and checkpoint_config.get("dim") == 1536
                and checkpoint_config.get("num_heads") == 12)):
        return "wan2.1-t2v-1.3B"
    if (checkpoint_dir / "Wan2.1_VAE.pth").is_file():
        return "t2v-A14B"

    if model_type == "ti2v":
        return "ti2v-5B"
    if model_type == "t2v":
        return "t2v-A14B"

    if "ti2v" in directory_name and "5b" in directory_name:
        return "ti2v-5B"
    if "t2v" in directory_name and "a14b" in directory_name:
        return "t2v-A14B"

    raise ValueError(
        "Could not detect the Wan model variant from checkpoint directory "
        f"'{checkpoint_dir}'. Pass --model_variant t2v-A14B or "
        "--model_variant ti2v-5B or --model_variant "
        "wan2.1-t2v-1.3B explicitly.")


def resolve_model_variant(requested_variant, checkpoint_dir):
    if requested_variant != "auto":
        return requested_variant
    return detect_model_variant(checkpoint_dir)


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
    for name in ("low_noise_model", "high_noise_model", "model"):
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
    if hasattr(vae, "mean") and hasattr(vae, "std"):
        vae.mean = vae.mean.to(dtype=dtype, device=vae.device)
        vae.std = vae.std.to(dtype=dtype, device=vae.device)
        vae.scale = [vae.mean, 1.0 / vae.std]
    else:
        vae.scale = [
            value.to(dtype=dtype, device=vae.device)
            if isinstance(value, torch.Tensor) else value
            for value in vae.scale
        ]
    vae.model.to(device=vae.device, dtype=dtype)


def offload_vae_model(model):
    vae_model = getattr(getattr(model, "vae", None), "model", None)
    if vae_model is not None and next(vae_model.parameters()).device.type == "cuda":
        vae_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def onload_vae_model(model):
    vae = model.vae
    if next(vae.model.parameters()).device.type == "cpu":
        vae.model.to(device=vae.device, dtype=vae.dtype)
        if hasattr(vae, "mean") and hasattr(vae, "std"):
            vae.mean = vae.mean.to(device=vae.device, dtype=vae.dtype)
            vae.std = vae.std.to(device=vae.device, dtype=vae.dtype)
            vae.scale = [vae.mean, 1.0 / vae.std]


def onload_dit_models(model):
    for _, dit_model in unique_dit_models(model):
        if next(dit_model.parameters()).device.type == "cpu":
            dit_model.to(model.device)
    torch.cuda.empty_cache()


def make_model_config(wan_configs, model_variant):
    cfg = copy.deepcopy(wan_configs[model_variant])
    return cfg


def prompt_base_size_for_variant(model_variant):
    return MODEL_VARIANTS[model_variant]["prompt_base_size"]


def resolve_vae_decode_tile_size(model_variant, tile_height=None,
                                 tile_width=None):
    variant = MODEL_VARIANTS[model_variant]
    if tile_height is None:
        tile_height = variant["vae_decode_tile_height"]
    if tile_width is None:
        tile_width = variant["vae_decode_tile_width"]
    if tile_height is not None and tile_height <= 0:
        raise ValueError("VAE decode tile height must be positive.")
    if tile_width is not None and tile_width <= 0:
        raise ValueError("VAE decode tile width must be positive.")
    return tile_height, tile_width


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
                                   global_rope_threshold_x,
                                   max_relative_y=44,
                                   max_relative_x=79):
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
            block.self_attn.block_tiled_attn_max_relative_y = max_relative_y
            block.self_attn.block_tiled_attn_max_relative_x = max_relative_x

    for _, dit_model in unique_dit_models(model):
        set_model_block_tiling(dit_model)


def prepare_dit_for_timestep(model, timestep, boundary, offload_model):
    if boundary is not None:
        return model._prepare_model_for_timestep(timestep, boundary,
                                                 offload_model)

    dit_model = model.model
    if offload_model or model.init_on_cpu:
        if next(dit_model.parameters()).device.type == "cpu":
            dit_model.to(model.device)
    return dit_model


def guide_scale_for_timestep(guide_scale, timestep, boundary):
    if not isinstance(guide_scale, (tuple, list)):
        return guide_scale
    if len(guide_scale) != 2:
        raise ValueError("Guide scale sequences must contain exactly two values.")
    if boundary is None:
        if guide_scale[0] != guide_scale[1]:
            raise ValueError(
                "A single-DiT model requires one guide scale value.")
        return guide_scale[0]
    return guide_scale[1] if timestep.item() >= boundary else guide_scale[0]


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
    # WanModel expands this scalar timestep across seq_len. For prompt-only
    # TI2V-5B this is equivalent to the all-ones mask used by WanTI2V.t2v.
    timestep = torch.stack([timestep]).to(model.device)
    active_model = prepare_dit_for_timestep(
        model, timestep[0], boundary, offload_model)
    step_guide_scale = guide_scale_for_timestep(
        guide_scale, timestep[0], boundary)

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


def load_input_video(path, fps, max_frames=None):
    """Sample an input video at the output rate and return normalized CTHW frames."""
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Video input requires imageio. Install Wan2.2 requirements in "
            "the active environment.") from exc

    reader = imageio.get_reader(str(path))
    try:
        source_fps = float(reader.get_meta_data().get("fps") or fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"{path} has an invalid frame rate: {source_fps}.")
        if fps <= 0:
            raise ValueError("--fps must be positive.")
        frames = []
        next_sample = 0
        for source_index, frame in enumerate(reader):
            while next_sample * source_fps <= source_index * fps:
                if max_frames is not None and len(frames) >= max_frames:
                    break
                if frame.ndim != 3 or frame.shape[2] < 3:
                    raise ValueError(f"{path} must contain RGB video frames.")
                frames.append(torch.from_numpy(frame[..., :3].copy()))
                next_sample += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()

    count = 1 + (len(frames) - 1) // 4 * 4
    if count < 5:
        raise ValueError(
            f"{path} must provide at least five frames at {fps} FPS for "
            "Wan's 4n+1 temporal latent layout.")
    video = torch.stack(frames[:count]).permute(3, 0, 1, 2).float()
    video = video.div_(127.5).sub_(1.0)
    return video, source_fps


def encode_input_video(model, path, fps, max_frames):
    video, source_fps = load_input_video(path, fps, max_frames)
    _, frame_count, source_h, source_w = video.shape
    stride_h, stride_w = model.vae_stride[1:]
    latent_h = source_h // stride_h
    latent_w = source_w // stride_w
    if latent_h < 1 or latent_w < 1:
        raise ValueError(f"{path} has an unsupported aspect ratio.")
    encode_h = latent_h * stride_h
    encode_w = latent_w * stride_w
    if (source_h, source_w) != (encode_h, encode_w):
        video = F.interpolate(
            video.permute(1, 0, 2, 3),
            size=(encode_h, encode_w),
            mode="bilinear",
            align_corners=False).permute(1, 0, 2, 3)
    with torch.no_grad():
        latent = model.vae.encode([video.to(model.vae.device)])[0]
    del video
    expected_frames = (frame_count - 1) // model.vae_stride[0] + 1
    if latent.shape[1:] != (expected_frames, latent_h, latent_w):
        raise RuntimeError(
            f"Encoded video latent has shape {tuple(latent.shape)}; expected "
            f"(*, {expected_frames}, {latent_h}, {latent_w}).")
    return latent, {
        "input_frame_count": frame_count,
        "input_video_fps": source_fps,
        "input_video_size": f"{source_w}*{source_h}",
        "encoded_video_size": f"{encode_w}*{encode_h}",
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


def create_refinement_noise_latent(model, reference_size, target_size,
                                   frame_num):
    reference_width, reference_height = parse_size(reference_size)
    target_width, target_height = parse_size(target_size)
    target_area = target_width * target_height
    latent_h, latent_w = best_latent_size(
        reference_width, reference_height, target_area, model.vae_stride,
        model.patch_size)
    latent_frames = (frame_num - 1) // model.vae_stride[0] + 1
    latent = torch.randn(
        model.vae.model.z_dim,
        latent_frames,
        latent_h,
        latent_w,
        dtype=model.vae.dtype,
        device=model.device)
    seq_len = compute_seq_len(model, latent.shape)
    decoded_height = latent_h * model.vae_stride[1]
    decoded_width = latent_w * model.vae_stride[2]
    metadata = {
        "size": f"{decoded_width}*{decoded_height}",
        "latent_shape": tuple(latent.shape),
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
    model_boundary = getattr(model, "boundary", None)
    boundary = (
        model_boundary * model.num_train_timesteps
        if model_boundary is not None else None)
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
                                 stride_w=140,
                                 temporal_scale=4,
                                 spatial_scale=(8, 8)):
    
    vae_model = vae.model
    device = vae.device
    reflect_pad = 24

    z = latent.to(device).unsqueeze(0)
    _, _, latent_frames, latent_h, latent_w = z.shape
    if tile_h is not None and tile_h <= 0:
        raise ValueError("VAE decode tile height must be positive.")
    if tile_w is not None and tile_w <= 0:
        raise ValueError("VAE decode tile width must be positive.")
    tile_h = latent_h if tile_h is None else min(tile_h, latent_h)
    stride_h = tile_h if stride_h is None else min(stride_h, tile_h)
    tile_w = latent_w if tile_w is None else min(tile_w, latent_w)
    stride_w = tile_w if stride_w is None else min(stride_w, tile_w)
    blend_h = tile_h - stride_h
    blend_w = tile_w - stride_w
    scale_h, scale_w = spatial_scale
    out_frames = _vae_decoded_frame_count(latent_frames, temporal_scale)
    out_h, out_w = latent_h * scale_h, latent_w * scale_w


    with torch.no_grad(), torch.amp.autocast("cuda", dtype=vae.dtype):

        z, (pad_h, pad_w) = _reflect_pad_spatial_5d(z, reflect_pad, reflect_pad)
        tasks = _vae_tile_tasks(latent_h, latent_w, tile_h, tile_w, stride_h, stride_w)
        if is_main_process():
            logging.info(
                "Decoding latent shape %s with %d VAE tiles; maximum padded "
                "tile is %dx%d latent positions",
                tuple(latent.shape),
                len(tasks),
                tile_h + 2 * pad_h,
                tile_w + 2 * pad_w)

        values = torch.zeros(1, 3, out_frames, out_h, out_w, dtype=torch.float32)
        weights = torch.zeros(1, 1, out_frames, out_h, out_w, dtype=torch.float32)

        for y0, y1, x0, x1 in tqdm(tasks, desc="VAE Decode", unit="tile",
                                    disable=not is_main_process()):
            tile = z[:, :, :, y0:y1 + 2 * pad_h, x0:x1 + 2 * pad_w].to(device)
            tile_out = vae_model.decode(tile, vae.scale).float().cpu()
            core_h = (y1 - y0) * scale_h
            core_w = (x1 - x0) * scale_w
            tile_out = tile_out[
                ..., pad_h * scale_h:pad_h * scale_h + core_h,
                pad_w * scale_w:pad_w * scale_w + core_w]

            out_y0, out_x0 = y0 * scale_h, x0 * scale_w
            mask = _linear_blend_mask(
                tile_out,
                is_bound=(y0 == 0, y1 >= latent_h, x0 == 0, x1 >= latent_w),
                border_width=(blend_h * scale_h,
                              blend_w * scale_w)).float().cpu()
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
                            tile_w=128,
                            temporal_scale=4,
                            spatial_scale=(8, 8)):
    
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Saving videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc
    
    video = decode_latent_to_video_tiled(
        vae,
        latent,
        tile_h=tile_h,
        tile_w=tile_w,
        temporal_scale=temporal_scale,
        spatial_scale=spatial_scale)
    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in video.unbind(1):
            frame = ((frame.float() + 1.0) * 127.5).clamp_(0, 255)
            frame = frame.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            writer.append_data(frame)
    finally:
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


def create_vae(cfg, model_variant, checkpoint_dir, dtype, device):
    vae_path = os.path.join(checkpoint_dir, cfg.vae_checkpoint)
    if MODEL_VARIANTS[model_variant]["vae_version"] == "2.2":
        from wan.modules.vae2_2 import Wan2_2_VAE

        return Wan2_2_VAE(vae_pth=vae_path, dtype=dtype, device=device)

    from wan.modules.vae2_1 import Wan2_1_VAE

    return Wan2_1_VAE(vae_pth=vae_path, dtype=dtype, device=device)


def decode_latent_only(args, cfg):
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", args.device_id))
    if rank != 0:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    latent_path = Path(args.decode_latent)
    output_path = latent_path.with_suffix(".mp4")
    latent, metadata = load_latent_payload(
        latent_path, args.decode_latent_key)
    saved_variant = metadata.get("model_variant")
    if saved_variant is not None and saved_variant != args.model_variant:
        raise ValueError(
            f"Latent was generated with {saved_variant}, but checkpoint/model "
            f"selection resolved to {args.model_variant}.")
    fps = metadata.get("fps", args.fps)
    vae_dtype = parse_torch_dtype(args.vae_dtype)
    vae = create_vae(
        cfg,
        args.model_variant,
        args.ckpt_dir,
        vae_dtype,
        torch.device(f"cuda:{local_rank}"))
    set_vae_dtype(vae, vae_dtype)
    tile_h, tile_w = resolve_vae_decode_tile_size(
        args.model_variant,
        args.vae_decode_tile_height,
        args.vae_decode_tile_width)
    logging.info(
        "Using VAE decode tile height=%s, width=%s latent positions",
        "full" if tile_h is None else tile_h,
        "full" if tile_w is None else tile_w)
    if metadata.get("decoded_latent_key") == "prompt_base_latent":
        with torch.no_grad():
            video = vae.decode([latent.to(vae.device)])[0]
        save_video_tensor(video, output_path, fps)
    else:
        save_latent_video_tiled(
            vae,
            latent,
            output_path,
            fps,
            tile_h=tile_h,
            tile_w=tile_w,
            temporal_scale=cfg.vae_stride[0],
            spatial_scale=cfg.vae_stride[1:])

    print(f"Saved video as {output_path}", flush=True)


def run(args, model, cfg, prompt, output_latent, prompt_index):

    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    sample_steps = (
        args.sample_steps if args.sample_steps is not None else cfg.sample_steps)
    sample_shift = (
        args.sample_shift if args.sample_shift is not None else cfg.sample_shift)
    guide_scale = (
        args.sample_guide_scale
        if args.sample_guide_scale is not None else cfg.sample_guide_scale)
    step_offload_model = args.offload_model and not args.dit_fsdp
    text_offload_model = args.offload_model and not args.t5_fsdp

    if args.input_video is None and frame_num % 4 != 1:
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
    prompt_base_size = (
        args.prompt_base_size
        if args.prompt_base_size is not None
        else prompt_base_size_for_variant(args.model_variant))
    ar_max_relative_y = MODEL_VARIANTS[
        args.model_variant]["ar_max_relative_y"]
    ar_max_relative_x = MODEL_VARIANTS[
        args.model_variant]["ar_max_relative_x"]

    context = context_null = None
    restart_scheduler = restart_timesteps = restart_sigmas = None
   
    context, context_null = prepare_text_context(
        model, prompt, args.negative_prompt, text_offload_model)
    restart_scheduler, restart_timesteps, restart_sigmas = make_dpmpp_scheduler(
        model, sample_steps, sample_shift)
    if args.offload_model and (args.t5_fsdp or args.dit_fsdp):
        onload_dit_models(model)

    def configure_block_tiled_attention(enabled):
        set_block_tiled_self_attention(
            model,
            enabled,
            args.block_tiled_self_attn_tile_height,
            args.block_tiled_self_attn_tile_width,
            global_rope_threshold_y,
            global_rope_threshold_x,
            ar_max_relative_y,
            ar_max_relative_x)

    full_restart = args.round_noise_steps == sample_steps
    video_metadata = {}
    if args.input_video is not None:
        if full_restart:
            raise ValueError(
                "--round_noise_steps must be less than sample_steps when "
                "--input_video is provided; a full restart discards the video.")
        if args.offload_model:
            offload_dit_models(model)
            onload_vae_model(model)
        base_latent, video_metadata = encode_input_video(
            model, args.input_video, args.fps, frame_num)
        if is_main_process():
            logging.info(
                "Encoded %d input frames from %s at %s, then "
                "latent-resizing to target area %s",
                video_metadata["input_frame_count"], args.input_video,
                video_metadata["encoded_video_size"], encode_size)
        prompt_base_latent_cpu = (
            base_latent.detach().cpu() if is_main_process() else None)
        clean_latent, resize_metadata = resize_latent_to_size(
            model, base_latent,
            video_metadata["encoded_video_size"], encode_size)
        seq_len = compute_seq_len(model, clean_latent.shape)
        prompt_base_latent_shape = tuple(base_latent.shape)
        input_resize_mode = "latent"
        del base_latent
    elif full_restart:
        refinement_start_index = start_index_for_step_count(
            args.round_noise_steps, sample_steps)
        clean_latent, seq_len, resize_metadata = (
            create_refinement_noise_latent(
                model, prompt_base_size, encode_size, frame_num))
        prompt_base_latent_cpu = None
        prompt_base_latent_shape = None
        input_resize_mode = "direct_noise"
        if is_main_process():
            logging.info(
                "Skipping prompt-base generation because round_noise_steps "
                "equals sample_steps (%d); initialized target latent at %s "
                "directly from noise",
                sample_steps,
                resize_metadata["size"])
    else:
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
            global_rope_threshold_x,
            ar_max_relative_y,
            ar_max_relative_x)
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
        prompt_base_latent_shape = tuple(base_latent.shape)
        input_resize_mode = "latent"
        del base_latent
        if is_main_process():
            logging.info(
                "Generated prompt base at %s, then latent-resized to %s",
                prompt_base_size, resize_metadata["size"])

    metadata = {
        **resize_metadata,
        "input_mode": "video" if args.input_video is not None else "prompt",
        "input_video": str(Path(args.input_video).resolve()) if args.input_video else None,
        **video_metadata,
        "prompt_base_size": prompt_base_size,
        "prompt_base_latent_shape": prompt_base_latent_shape,
        "input_resize_mode": input_resize_mode,
        "input_frame_count": video_metadata.get("input_frame_count", frame_num),
        "model_variant": args.model_variant,
        "block_tiled_self_attn": args.block_tiled_self_attn,
        "block_tiled_self_attn_tile_size": (
            args.block_tiled_self_attn_tile_height,
            args.block_tiled_self_attn_tile_width),
        "block_tiled_self_attn_global_rope_threshold": (
            global_rope_threshold_y, global_rope_threshold_x),
        "block_tiled_self_attn_max_relative": (
            ar_max_relative_y, ar_max_relative_x),
    }
    if full_restart:
        metadata["base_generation_skipped"] = True

    if args.offload_model:
        offload_vae_model(model)

    configure_block_tiled_attention(args.block_tiled_self_attn)

    @contextmanager
    def noop_no_sync():
        yield

    final_latent = clean_latent.detach()
    with ExitStack() as no_sync_stack:
        for _, dit_model in unique_dit_models(model):
            no_sync = getattr(dit_model, "no_sync", noop_no_sync)
            no_sync_stack.enter_context(no_sync())
        if full_restart:
            first_sigma = float(restart_sigmas[0].item())
            if not math.isclose(first_sigma, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(
                    "Full-restart refinement requires the first scheduler "
                    f"sigma to be 1.0, but received {first_sigma}.")
            start_index = refinement_start_index
            final_latent, _ = denoise_trajectory(
                model=model,
                start_latent=final_latent,
                context=context,
                context_null=context_null,
                seq_len=seq_len,
                scheduler=restart_scheduler,
                timesteps=restart_timesteps,
                sigmas=restart_sigmas,
                sample_steps=sample_steps,
                start_index=start_index,
                guide_scale=guide_scale,
                offload_model=step_offload_model,
                y=None)
        else:
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
        description="Run Wan2.1 or Wan2.2 CineScale for every prompt in a "
        "JSON file, optionally refining an input video."
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
        "--input_video",
        default=None,
        help="Video to encode with the selected Wan VAE instead of generating "
        "a prompt base. Reused for each prompt in --prompts_json.")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Wan2.2 T2V-A14B, Wan2.2 TI2V-5B, or Wan2.1 T2V-1.3B "
        "checkpoint directory.")
    parser.add_argument(
        "--model_variant",
        type=parse_model_variant,
        default="auto",
        metavar="{auto,t2v-A14B,ti2v-5B,wan2.1-t2v-1.3B}",
        help="Model architecture to load. By default, detect it from the "
        "checkpoint layout.")
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
    parser.add_argument(
        "--vae_decode_tile_height",
        type=int,
        default=None,
        help="VAE spatial decode tile height in latent positions. Defaults "
        "to 64 for Wan2.1 1.3B and full height for other variants.")
    parser.add_argument(
        "--vae_decode_tile_width",
        type=int,
        default=None,
        help="VAE spatial decode tile width in latent positions. Defaults "
        "to 64 for Wan2.1 1.3B and 128 for other variants.")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output FPS stored with the latent. Defaults to the model config.")
    parser.add_argument(
        "--size",
        default="3840*2160",
        help="Target area as width*height. The prompt base or input video's "
        "aspect ratio is preserved during latent upsampling.")
    parser.add_argument(
        "--prompt_base_size",
        default=None,
        help="Override the model variant's prompt-base resolution as "
        "width*height. For example, use 1280*720 to experimentally generate "
        "a 720p Wan2.1-1.3B base before high-resolution refinement.")
    parser.add_argument(
        "--frame_num", type=int, default=None,
        help="Prompt mode frame count. In video mode, maximum input frames "
        "sampled at --fps (defaults to the model config and is trimmed "
        "to 4n+1).")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--round_noise_steps",
        type=int,
        default=30,
        help="Exact high-resolution refinement steps. When this equals "
        "sample_steps, prompt-base generation and latent resizing are "
        "skipped and the target latent is initialized directly from noise.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Noise schedule shift. Defaults to the selected model config.")
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--block_tiled_self_attn",
        type=str2bool,
        default=True,
        help="Tile only DiT self-attention inside each block, stitch the self-attention output, then run global cross-attention/FFN.")
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


def create_generation_model(wan, args, cfg):
    pipeline_name = MODEL_VARIANTS[args.model_variant]["pipeline"]
    pipeline_class = getattr(wan, pipeline_name)
    return pipeline_class(
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


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import wan
    from wan.configs import WAN_CONFIGS

    args.model_variant = resolve_model_variant(
        args.model_variant, args.ckpt_dir)
    cfg = make_model_config(WAN_CONFIGS, args.model_variant)
    if args.fps is None:
        args.fps = cfg.sample_fps
    args.model_version = MODEL_VARIANTS[args.model_variant]["model_version"]
    logging.info("Using Wan%s %s checkpoint layout", args.model_version,
                 args.model_variant)
    if args.decode_latent is not None:
        decode_latent_only(args, cfg)
        return
    if args.input_video is not None and not Path(args.input_video).is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input_video}")
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
    seed = args.base_seed if args.base_seed >= 0 else (
        random.randint(0, sys.maxsize) if rank == 0 else 0)
    if dist.is_initialized():
        seed_holder = [seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_holder, src=0)
        seed = seed_holder[0]
    random.seed(seed)
    torch.manual_seed(seed)
    args.run_seed = seed

    model = create_generation_model(wan, args, cfg)
    model.model_version = args.model_version
    model.model_variant = args.model_variant

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


