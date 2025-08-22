import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download
import torch.distributed as dist

# Download models
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")

# Load models
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

model_manager.load_lora("models/t2v_1.3b_ntk20.ckpt", lora_alpha=1.0)

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
from xfuser.core.distributed import (initialize_model_parallel,
                                     init_distributed_environment)
init_distributed_environment(
    rank=dist.get_rank(), world_size=dist.get_world_size())

# initialize_model_parallel(
#     sequence_parallel_degree=dist.get_world_size(),
#     ring_degree=1,
#     ulysses_degree=dist.get_world_size(),
# )
initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=dist.get_world_size(),
    ulysses_degree=1,
)
torch.cuda.set_device(dist.get_rank())

pipe = WanVideoPipeline.from_model_manager(model_manager, 
                                           torch_dtype=torch.bfloat16, 
                                           device=f"cuda:{dist.get_rank()}", 
                                           use_usp=True if dist.get_world_size() > 1 else False)
pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

height=1088
width=1920
pipe.dit.set_ntk([1, 20, 20])

# Text-to-video
video = pipe(
    prompt="In a serene forest clearing, a majestic owl with striking amber eyes perches on a gloved hand, its feathers a blend of tawny and cream hues. The man, wearing a rugged leather jacket and a wide-brimmed hat, gently gestures with his other hand, guiding the owl's gaze. Sunlight filters through the canopy, casting dappled patterns on the forest floor. The owl spreads its wings, revealing intricate patterns, as it prepares to take flight. The man, with a calm and focused demeanor, watches intently, embodying a deep bond of trust and understanding between human and bird in this tranquil woodland setting.",
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    num_inference_steps=50,
    seed=123, tiled=True,
    height=height, 
    width=width,
    num_frames=81,
    cfg_scale=5.0,
    sigma_shift=7.0,
)
if dist.get_rank() == 0:
    save_video(video, "video_w{}.mp4".format(width), fps=15)
