from unittest.mock import AsyncMock, Mock

import numpy as np
import numpy.typing as npt
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)

import env
import model_provider
from my_logging import logger


def mock_StableDiffusionPipeline() -> Mock:
    pipe_spec = (
        "tokenizer to __call__ vae text_encoder tokenizer unet scheduler safety_checker feature_extractor".split()
    )
    pipe = Mock(spec_set=pipe_spec)
    pipe.to.return_value = pipe
    pipe.return_value = StableDiffusionPipelineOutput(
        images=np.zeros((1, env.WIDTH, env.HEIGHT, 3), dtype=np.float16), nsfw_content_detected=[False]
    )
    return pipe


class _FakeModelProvider(model_provider.ModelProvider):
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
        assert False


def mock_ModelProvider() -> Mock:
    p = AsyncMock(spec=_FakeModelProvider())
    p.return_value = np.zeros((1, env.WIDTH, env.HEIGHT, 3), dtype=np.float16)
    return p
