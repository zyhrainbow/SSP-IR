import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel

from pipelines.pipeline_bim import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
clip_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel
    from models.model import TPBNet

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.ours_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.ours_model_path, subfolder="controlnet")
    refine = TPBNet(last_dim=512)
    ckpt_path = os.path.join(args.ours_model_path, "refine")+'/refine.pt'
    state_dict = torch.load(ckpt_path, map_location="cpu")
    refine.load_state_dict(state_dict, strict=True)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    refine.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    refine.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline, refine, weight_dtype

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, image_name, device='cuda',image_encoder=None):
    validation_prompt = ""
    # import sys
    # sys.path.append("/media/ps/ssd2/zyh/lv/SUPIR")
    from llava.llava_agent import LLavaAgent
    from CKPT_PTH import LLAVA_MODEL_PATH
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=device)
   
    image0 = Image.open(image_name).convert("RGB")
    captions = llava_agent.gen_image_caption([image0])[0]
    print(captions)
    

    
    
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = clip_transforms(lq)


    return validation_prompt, lq#ram_encoder_hidden_states #

def main(args, enable_xformers_memory_efficient_attention=True,):


    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline, refine, weight_dtype = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    # model = load_tag_model(args, accelerator.device)
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device, dtype=weight_dtype)
    
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")
            # base_name = os.path.basename(image_name).split('.')[0]
            # prompt_path = os.path.join('/media/ps/ssd2/zyh/lv/emotion/SeeSR/preset/datasets/llm@20240304/reallr200-noaddprompt/txt', base_name+'.txt')
            # file = open(prompt_path, 'r')
            # validation_prompt = file.read()
            # file.close()
            validation_prompt, clip_values = get_validation_prompt(args, validation_image, image_name, image_encoder=clip_model) #, ram_en