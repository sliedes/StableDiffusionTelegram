import os

from dotenv import load_dotenv
import torch

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")

MODEL_DATA = os.getenv("MODEL_DATA", "CompVis/stable-diffusion-v1-4")
LOW_VRAM_MODE = os.getenv("LOW_VRAM", "true").lower() == "true"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", False)
SAFETY_CHECKER = os.getenv("SAFETY_CHECKER", "true").lower() == "true"
HEIGHT = int(os.getenv("HEIGHT", "512"))
WIDTH = int(os.getenv("WIDTH", "512"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "100"))
STRENGTH = float(os.getenv("STRENGTH", "0.75"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
ADMIN_ID = int(os.getenv("ADMIN_ID"))
CHAT_ID = int(os.getenv("CHAT_ID"))
ADMIN_ONLY = os.getenv("ADMIN_ONLY", "false").lower() == "true"

MODEL_REVISION = "fp16" if LOW_VRAM_MODE else None
TORCH_DTYPE = torch.float16 if LOW_VRAM_MODE else None
