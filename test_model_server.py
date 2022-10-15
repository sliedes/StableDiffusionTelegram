#!/usr/bin/env python3

from typing import Any, AsyncIterable, Optional
from unittest.mock import ANY, Mock

import numpy as np
import pytest
from grpc.aio import ServicerContext
from pytest_mock import MockerFixture

import env
import model_server
from model_server import ModelServicer
from model_server_pb2 import Image, ImGenRequest, ImGenRequestMetadata, TokenizeRequest
from my_logging import logger

_ImGenRequestMetadata_template = ImGenRequestMetadata(
    width=env.WIDTH,
    height=env.HEIGHT,
    seed=1234,
    iterations=env.NUM_INFERENCE_STEPS,
    strength=env.STRENGTH,
    guidance_scale=env.GUIDANCE_SCALE,
)

_Image_template = Image(width=env.WIDTH, height=env.HEIGHT, dtype="B", data=b"x" * (env.WIDTH * env.HEIGHT * 3))


def make_ImGenRequest(**kwargs: Any) -> ImGenRequest:
    meta = ImGenRequestMetadata()
    meta.CopyFrom(_ImGenRequestMetadata_template)
    for k, v in kwargs.items():
        setattr(meta, k, v)
    return ImGenRequest(req_metadata=meta)


@logger.catch
@pytest.fixture
async def servicer(mocker: MockerFixture) -> AsyncIterable[ModelServicer]:
    pipe_spec = (
        "tokenizer to __call__ vae text_encoder tokenizer unet scheduler safety_checker feature_extractor".split()
    )

    txt2img_pipe = Mock(spec=pipe_spec)
    txt2img_pipe.to.return_value = txt2img_pipe
    mocker.patch("model_server.StableDiffusionPipeline.from_pretrained", return_value=txt2img_pipe)

    img2img_pipe = Mock(spec=pipe_spec)
    img2img_pipe.to.return_value = img2img_pipe
    mocker.patch("model_server.StableDiffusionImg2ImgPipeline", return_value=img2img_pipe)

    servicer = ModelServicer(model_server.GPUWorker())
    await servicer.worker._start_worker_task
    yield servicer
    await servicer.worker.stop()


async def test_tokenize(servicer: ModelServicer) -> None:
    servicer.worker._a_pipe.tokenizer.tokenize.return_value = ["foo"]
    resp = await servicer.tokenize_prompt(TokenizeRequest(prompt="Hello, world!"), context=Mock(spec=ServicerContext))
    assert resp.prompt_tokens == ["foo"]


def check_image(im: Image, expect_dims: Optional[int] = None) -> None:
    if expect_dims:
        assert im.width == expect_dims
        assert im.height == expect_dims
    assert im.dtype == "B"
    assert len(im.data) == im.width * im.height * 3 * 2


async def test_txt2img_compute(servicer: ModelServicer) -> None:
    req = make_ImGenRequest(prompt="A ball on a floor", test_no_compute=False, iterations=2)
    servicer.worker._a_pipe.return_value = {"sample": [np.zeros((env.WIDTH, env.HEIGHT, 3), dtype=np.float16)]}
    resp = await servicer.generate_image(req, context=Mock(spec=ServicerContext))
    servicer.worker._a_pipe.assert_called_once_with(
        **dict(
            prompt=[req.req_metadata.prompt],
            generator=ANY,
            width=req.req_metadata.width,
            height=req.req_metadata.height,
            strength=req.req_metadata.strength,
            guidance_scale=req.req_metadata.guidance_scale,
            num_inference_steps=req.req_metadata.iterations,
            output_type="numpy",
        )
    )
    check_image(resp.image, env.WIDTH)
    assert resp.req_metadata == req.req_metadata


async def test_txt2img_nocompute(servicer: ModelServicer) -> None:
    req = make_ImGenRequest(prompt="A ball on a floor", test_no_compute=True, iterations=2)
    resp = await servicer.generate_image(req, context=Mock(spec=ServicerContext))
    servicer.worker._a_pipe.assert_not_called()
    assert resp.req_metadata == req.req_metadata
    check_image(resp.image, 0)


async def test_img2img_compute(servicer: ModelServicer) -> None:
    req = make_ImGenRequest(prompt="A ball on a floor", test_no_compute=False, iterations=2)
    req.image.CopyFrom(_Image_template)
    servicer.worker._a_img2imgpipe.return_value = {"sample": [np.zeros((env.WIDTH, env.HEIGHT, 3), dtype=np.float16)]}
    resp = await servicer.generate_image(req, context=Mock(spec=ServicerContext))
    servicer.worker._a_img2imgpipe.assert_called_once_with(
        **dict(
            prompt=[req.req_metadata.prompt],
            init_image=ANY,
            generator=ANY,
            strength=req.req_metadata.strength,
            guidance_scale=req.req_metadata.guidance_scale,
            num_inference_steps=req.req_metadata.iterations,
            output_type="numpy",
        )
    )
    check_image(resp.image, env.WIDTH)
