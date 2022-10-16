from unittest.mock import Mock

import numpy as np
import PIL

import env


def mock_StableDiffusionPipeline() -> Mock:
    pipe_spec = (
        "tokenizer to __call__ vae text_encoder tokenizer unet scheduler safety_checker feature_extractor".split()
    )
    pipe = Mock(spec=pipe_spec)
    pipe.to.return_value = pipe
    pipe.return_value = {
        "sample": [PIL.Image.new("RGB", (env.WIDTH, env.HEIGHT))]
    }  # [np.zeros((env.WIDTH, env.HEIGHT, 3), dtype=np.float16)]}
    return pipe
