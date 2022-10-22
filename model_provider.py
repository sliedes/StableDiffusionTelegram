from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ModelProvider(ABC):
    """Abstract class for Stable Diffusion model providers.
    Implementations can provide the model locally or remotely."""

    @abstractmethod
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
        """Generate an image from an image prompt anf an optional initial image.

        Args:
            prompt (str): Image prompt
            width (int): Width of the generated image
            height (int): Height of the generated image
            seed (int): Random seed
            strength (float): Diffusion strength
            guidance_scale (float): Guidance scale
            num_inference_steps (int): Number of inference steps
            initial_image (npt.NDArray[np.float_] | None): Initial image

        Returns:
            npt.NDArray[np.float_]: Generated image
        """
        raise NotImplementedError
