import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download
import torch.distributed as dist
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Download models
snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-720P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)

model_manager.load_lora("models/i2v_14b_ntk20.ckpt", lora_alpha=1.0)

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
from xfuser.core.distributed import (initialize_model_parallel,
                                     init_distributed_environment)
init_distributed_environment(
    rank=dist.get_rank(), world_size=dist.get_world_size())

initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=1,
    ulysses_degree=dist.get_world_size(),
)
torch.cuda.set_device(dist.get_rank())

pipe = WanVideoPipeline.from_model_manager(model_manager, 
                                           torch_dtype=torch.bfloat16, 
                                           device=f"cuda:{dist.get_rank()}", 
                                           use_usp=True if dist.get_world_size() > 1 else False)
pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

height=2176
width=3840
pipe.dit.set_ntk([1, 20, 20])
# May set attention_coef to 1.5 for better results (line 123, diffsynth/distributed/xdit_context_parallel.py)

prompt = "a brown bear in the water with a fish in its mouth"
image = Image.open("./assets/a brown bear in the water with a fish in its mouth.jpg")

# Image-to-video
video = pipe(
    prompt = prompt,
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    input_image=image,
    num_inference_steps=50,
    seed=0, tiled=True,
    height=height, 
    width=width,
    num_frames=81,
    cfg_scale=5.0,
    sigma_shift=5.0,
)
if dist.get_rank() == 0:
    save_video(video, "video_i2v_w{}.mp4".format(width), fps=15)
