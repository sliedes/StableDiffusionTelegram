import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar, cast

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
        logger.info("Saved text2img.pt")
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


F = TypeVar("F", bound=Callable[..., Any])


def _with_autocast(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with autocast("cuda"):
            return func(*args, **kwargs)

    return cast(F, wrapper)


class LocalModelProvider(ModelProvider):
    """Provides a Stable Diffusion model locally."""

    _gpu_lock: asyncio.Lock
    _txt2imgPipe: StableDiffusionPipeline
    _img2imgPipe: StableDiffusionImg2ImgPipeline

    def __init__(self) -> None:
        super().__init__()
        self._gpu_lock = asyncio.Lock()
        self._txt2imgPipe, self._img2imgPipe = load_models()

    async def __call__(
        self,
        prompt: str,
        seed: int,
        width: int = 512,
        height: int = 512,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        init_image: npt.NDArray[np.float_] | None = None,
    ) -> npt.NDArray[np.float_]:
        async with self._gpu_lock:
            logger.info("generate_image (init_image={}): {}", init_image is not None, prompt)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)

            if init_image is not None:
                image = (
                    await asyncio.to_thread(
                        _with_autocast(self._img2imgPipe),
                        prompt=[prompt],
                        init_image=cast(torch.FloatTensor, torch.as_tensor(init_image).to("cuda").float()),  # FIXME
                        generator=generator,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        output_type="np.array",
                    )
                ).images
            else:
                image = (
                    await asyncio.to_thread(
                        _with_autocast(self._txt2imgPipe),
                        prompt=[prompt],
                        generator=generator,
                        strength=strength,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        output_type="np.array",
                    )
                ).images
        assert isinstance(image, np.ndarray), type(image)
        return image


__all__ = ["LocalModelProvider"]
