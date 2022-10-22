import asyncio
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from diffusers import StableDiffusionPipeline as StableDiffusionPipeline
from torch import autocast as autocast  # type: ignore[attr-defined]

import env
from image_to_image import (
    StableDiffusionImg2ImgPipeline as StableDiffusionImg2ImgPipeline,
)
from model_provider import ModelProvider
from my_logging import logger

UPDATE_MODEL = False


# disable safety checker if wanted
def dummy_checker(images: Any, **kwargs: Any) -> tuple[Any, bool]:
    return images, False


def load_models() -> tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
    logger.info("Loading text2img pipeline")
    if UPDATE_MODEL:
        pipe = StableDiffusionPipeline.from_pretrained(
            env.MODEL_DATA,
            revision=env.MODEL_REVISION,
            torch_dtype=env.TORCH_DTYPE,
            use_auth_token=env.HF_AUTH_TOKEN,
        )
        torch.save(pipe, "text2img.pt")
    else:
        pipe = torch.load("text2img.pt")  # type: ignore[no-untyped-call]
    logger.info("Loaded text2img pipeline")

    img2imgPipe = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=pipe.safety_checker,
        feature_extractor=pipe.feature_extractor,
    )

    pipe = pipe.to("cuda")
    img2imgPipe = img2imgPipe.to("cuda")

    if not env.SAFETY_CHECKER:
        pipe.safety_checker = dummy_checker
        img2imgPipe.safety_checker = dummy_checker

    return pipe, img2imgPipe


class LocalModelProvider(ModelProvider):
    """Provides a Stable Diffusion model locally."""

    _gpu_lock: asyncio.Lock
    _txt2imgPipe: StableDiffusionPipeline
    _img2imgPipe: StableDiffusionImg2ImgPipeline

    def __init__(self) -> None:
        super().__init__()
        self._txt2imgPipe, self._img2imgPipe = load_models()
        self._gpu_lock = asyncio.Lock()

    async def __call__(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        init_image: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float_]:
        async with self._gpu_lock:
            logger.info("generate_image (init_image={}): {}", init_image is not None, prompt)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)

            if init_image is not None:
                with autocast("cuda"):
                    image = self._img2imgPipe(
                        prompt=[prompt],
                        init_image=cast(torch.FloatTensor, torch.as_tensor(init_image).to("cuda").float()),
                        generator=generator,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    )["sample"][0]
            else:
                with autocast("cuda"):
                    image = self._txt2imgPipe(
                        prompt=[prompt],
                        generator=generator,
                        strength=strength,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    )["sample"][0]
            return np.array(image)  # FIXME: avoid the PIL round trip?


__all__ = ["LocalModelProvider"]
