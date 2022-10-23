from unittest.mock import ANY, Mock

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

import local_model_provider
from mocks import mock_StableDiffusionPipeline


@pytest.fixture
def models(mocker: MockerFixture) -> None:
    mocker.patch("torch.load", return_value=mock_StableDiffusionPipeline())
    mocker.patch("torch.save")
    mocker.patch(
        "local_model_provider.StableDiffusionPipeline.from_pretrained", return_value=mock_StableDiffusionPipeline()
    )
    mocker.patch("local_model_provider.StableDiffusionImg2ImgPipeline", return_value=mock_StableDiffusionPipeline())


def test_construct_noupdate(mocker: MockerFixture, models: Mock) -> None:
    local_model_provider.LocalModelProvider()
    mocker.patch("local_model_provider.UPDATE_MODEL", new=False)
    torch.load.assert_called_once_with("text2img.pt")  # type: ignore[attr-defined]


def test_construct_update(mocker: MockerFixture, models: Mock) -> None:
    mocker.patch("local_model_provider.UPDATE_MODEL", new=True)
    local_model_provider.LocalModelProvider()
    torch.save.assert_called_once_with(ANY, "text2img.pt")  # type: ignore[attr-defined]


@pytest.fixture
def provider(mocker: MockerFixture, models: Mock) -> local_model_provider.LocalModelProvider:
    return local_model_provider.LocalModelProvider()


async def test_txt2img(provider: local_model_provider.LocalModelProvider) -> None:
    ret = await provider(
        prompt="Hello",
        width=512,
        height=512,
        seed=1234,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=50,
        init_image=None,
    )
    assert isinstance(ret, np.ndarray)
    provider._txt2imgPipe.assert_called()
    provider._img2imgPipe.assert_not_called()


async def test_img2img(provider: local_model_provider.LocalModelProvider) -> None:
    ret = await provider(
        prompt="Hello",
        width=512,
        height=512,
        seed=1234,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=50,
        init_image=np.zeros((16, 16, 3)),
    )
    assert isinstance(ret, np.ndarray)
    provider._txt2imgPipe.assert_not_called()
    provider._img2imgPipe.assert_called()


async def test_returns_4d(provider: local_model_provider.LocalModelProvider) -> None:
    ret = await provider(
        prompt="Hello",
        width=512,
        height=512,
        seed=1234,
        strength=0.7,
        guidance_scale=7.5,
        num_inference_steps=50,
        init_image=None,
    )
    assert ret.shape == (1, 512, 512, 3)
