import random
from unittest.mock import Mock

import numpy as np
import PIL
import pytest
import telegram
import torch
from pytest import FixtureRequest
from pytest_mock import MockerFixture

import bot
import env
from mocks import mock_StableDiffusionPipeline


@pytest.fixture
def txt2img() -> Mock:
    return mock_StableDiffusionPipeline()


def make_photo() -> bytes:
    return bot.image_to_bytes(PIL.Image.new("RGB", (env.WIDTH, env.HEIGHT)))


@pytest.fixture(params=[False, True])
def maybe_photo(request: FixtureRequest) -> bytes | None:
    if request.param:
        return make_photo()
    return None


img2img = txt2img


def test_image_to_bytes() -> None:
    im = PIL.Image.new("RGB", (3, 3))
    im_bytes = bot.image_to_bytes(im)
    assert isinstance(im_bytes, bytes)


def test_get_try_again_markup() -> None:
    markup = bot.get_try_again_markup()
    assert isinstance(markup, telegram.InlineKeyboardMarkup)


def test_parse_seed() -> None:
    assert bot.parse_seed("seed:123 text") == (123, "text")

    # most importantly, these should not crash
    assert bot.parse_seed("seed:123") == (None, "seed:123")
    assert bot.parse_seed("seed:xyz text") == (None, "text")


async def test_generate_image_no_seed(txt2img: Mock, img2img: Mock, maybe_photo: bytes | None) -> None:
    im, seed, prompt = await bot.generate_image(txt2img, img2img, "Hello, world!", photo=maybe_photo)
    assert isinstance(im, np.ndarray)
    assert im.shape == (env.WIDTH, env.HEIGHT, 3)
    assert isinstance(seed, int)
    assert prompt == "Hello, world!"


async def test_generate_image_with_seed(txt2img: Mock, img2img: Mock, maybe_photo: bytes | None) -> None:
    im, seed, prompt = await bot.generate_image(txt2img, img2img, "seed:1234 Hello, world!", photo=maybe_photo)
    assert isinstance(im, np.ndarray)
    assert im.shape == (env.WIDTH, env.HEIGHT, 3)
    assert seed == 1234
    assert prompt == "Hello, world!"


async def test_generate_image_ignore_seed(
    txt2img: Mock, img2img: Mock, mocker: MockerFixture, maybe_photo: bytes | None
) -> None:
    mocker.patch("random.randint")
    random.randint.return_value = 42  # type: ignore[attr-defined]
    im, seed, prompt = await bot.generate_image(
        txt2img, img2img, "seed:1234 Hello, world!", ignore_seed=True, photo=maybe_photo
    )
    assert isinstance(im, np.ndarray)
    assert im.shape == (env.WIDTH, env.HEIGHT, 3)
    assert seed == 42
    assert prompt == "Hello, world!"


@pytest.fixture
def update() -> Mock:
    update = Mock(spec=telegram.Update)
    update.effective_user.id = 0
    update.effective_chat.id = env.CHAT_ID
    return update


def test_perms_ok_admin(update: Mock) -> None:
    update.effective_user.id = env.ADMIN_ID
    update.effective_chat.id = 0
    assert bot.perms_ok(update)


def test_perms_ok_not_admin(update: Mock) -> None:
    update.effective_user.id = 0
    update.effective_chat.id = 0
    assert not bot.perms_ok(update)


def test_perms_ok_chat(update: Mock) -> None:
    update.effective_user.id = 0
    update.effective_chat.id = env.CHAT_ID
    assert bot.perms_ok(update)


def test_extract_query_from_string() -> None:
    assert bot.extract_query_from_string("hello") == "hello"
    assert bot.extract_query_from_string(bot.COMMAND + "hello") == "hello"
    assert bot.extract_query_from_string(bot.COMMAND * 3 + "hello") == "hello"
    assert bot.extract_query_from_string(bot.COMMAND + '"hello "quotes"" (Seed: 1)') == 'hello "quotes"'
    assert bot.extract_query_from_string(bot.COMMAND + '"hello" (Seed: 123)') == "hello"


@pytest.fixture(params=[False, True])
def tryagain(request: FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def message() -> Mock:
    message = Mock(spec=telegram.Message)
    message.text = "hello"
    message.photo = []
    message.reply_to_message = None
    message.caption = None
    return message


message2 = message


async def test_parse_request_no_command_no_photo(update: Mock, message: Mock, tryagain: bool) -> None:
    message.text = "hello"
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is None


async def test_parse_request_with_command(update: Mock, message: Mock, tryagain: bool) -> None:
    message.text = bot.COMMAND + "hello"
    message.photo = []
    message.reply_to_message = None
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is not None
    assert request.prompt == "hello"
    assert request.photo is None
    assert request.tryagain == tryagain


@pytest.fixture
def photo() -> list[Mock]:
    photosize = Mock(spec=telegram.PhotoSize)
    tg_file = Mock(spec=telegram.File)
    photosize.get_file.return_value = tg_file
    tg_file.download_as_bytearray.return_value = b"photo"
    return [photosize]


async def test_parse_request_photo_without_caption(
    update: Mock, message: Mock, photo: list[Mock], tryagain: bool
) -> None:
    message.photo = photo
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is None


async def test_parse_request_photo_with_caption(update: Mock, message: Mock, photo: list[Mock], tryagain: bool) -> None:
    message.caption = "hello"
    message.photo = photo
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is not None
    assert request.prompt == "hello"
    assert request.photo == b"photo"
    assert request.tryagain == tryagain


async def test_parse_request_photo_with_caption_and_command(
    update: Mock, message: Mock, photo: list[Mock], tryagain: bool
) -> None:
    message.caption = bot.COMMAND + "hello"
    message.photo = photo
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is not None
    assert request.prompt == "hello"
    assert request.photo == b"photo"
    assert request.tryagain == tryagain


async def test_parse_request_reply_to_photo_no_command(
    update: Mock, message: Mock, message2: Mock, photo: list[Mock], tryagain: bool
) -> None:
    message.reply_to_message = message2
    message2.photo = photo
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is None


async def test_parse_request_reply_to_photo_with_command(
    update: Mock, message: Mock, message2: Mock, photo: list[Mock], tryagain: bool
) -> None:
    message.reply_to_message = message2
    message2.photo = photo
    message.text = bot.COMMAND + "hello"
    request = await bot.parse_request(update, message, tryagain=tryagain)
    assert request is not None
    assert request.prompt == "hello"
    assert request.photo == b"photo"

    # This should always be false; we won't be able to access the photo in transitive replies, so we shouldn't
    # generate a Try Again button for it.
    assert request.tryagain == False


@pytest.fixture(params=["update", "noupdate"])
def bot_(mocker: MockerFixture, request: FixtureRequest, txt2img: Mock, img2img: Mock) -> bot.Bot:
    mocker.patch("torch.load")
    mocker.patch("torch.save")
    mocker.patch("bot.StableDiffusionPipeline")
    mocker.patch("bot.StableDiffusionImg2ImgPipeline")
    if request.param == "noupdate":
        mocker.patch("bot.UPDATE_MODEL", False)
    else:
        assert request.param == "update"
        mocker.patch("bot.UPDATE_MODEL", True)
        bot.StableDiffusionPipeline.from_pretrained.return_value = txt2img
        bot.StableDiffusionImg2ImgPipeline.return_value = img2img
    b = bot.Bot()
    if request.param == "update":
        bot.StableDiffusionPipeline.from_pretrained.assert_called_once()
        torch.load.assert_not_called()  # type: ignore[attr-defined]
        torch.save.assert_called_once()  # type: ignore[attr-defined]
    else:
        bot.StableDiffusionPipeline.from_pretrained.assert_not_called()
        torch.load.assert_called_once()  # type: ignore[attr-defined]
        torch.save.assert_not_called()  # type: ignore[attr-defined]
    bot.StableDiffusionImg2ImgPipeline.assert_called_once()
    return b


def test_construct_bot(bot_: bot.Bot) -> None:
    pass
