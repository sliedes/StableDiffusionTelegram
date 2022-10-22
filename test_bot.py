import random
from dataclasses import dataclass
from typing import Callable, Protocol, cast
from unittest.mock import ANY, Mock

import numpy as np
import PIL
import pytest
import telegram
from pytest import FixtureRequest
from pytest_mock import MockerFixture

import bot
import env
from local_model_provider import LocalModelProvider
from mock_utils import assert_called_once_with_partial, fill_default_args
from mocks import mock_ModelProvider
from model_provider import ModelProvider
from my_logging import logger

# FIXME: Lots of stuff in this file also tests LocalModelProvider; that should be done separately.


_PHOTO_BYTES = bot.image_to_bytes(PIL.Image.new("RGB", (env.WIDTH, env.HEIGHT)))


@pytest.fixture(params=[False, True])
def maybe_photo(request: FixtureRequest) -> bytes | None:
    if request.param:
        return _PHOTO_BYTES
    return None


@pytest.fixture
def model_provider_() -> Mock:
    return mock_ModelProvider()


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


async def test_generate_image_no_seed(model_provider_: Mock, maybe_photo: bytes | None) -> None:
    im, seed, prompt = await bot.generate_image(model_provider_, "Hello, world!", photo=maybe_photo)
    assert isinstance(im, PIL.Image.Image)
    assert im.size == (env.WIDTH, env.HEIGHT)
    assert isinstance(seed, int)
    assert prompt == "Hello, world!"


async def test_generate_image_with_seed(model_provider_: Mock, maybe_photo: bytes | None) -> None:
    im, seed, prompt = await bot.generate_image(model_provider_, "seed:1234 Hello, world!", photo=maybe_photo)
    assert isinstance(im, PIL.Image.Image)
    assert im.size == (env.WIDTH, env.HEIGHT)
    assert seed == 1234
    assert prompt == "Hello, world!"


async def test_generate_image_ignore_seed(
    model_provider_: Mock, mocker: MockerFixture, maybe_photo: bytes | None
) -> None:
    mocker.patch("random.randint")
    random.randint.return_value = 42  # type: ignore[attr-defined]
    im, seed, prompt = await bot.generate_image(
        model_provider_, "seed:1234 Hello, world!", ignore_seed=True, photo=maybe_photo
    )
    assert isinstance(im, PIL.Image.Image)
    assert im.size == (env.WIDTH, env.HEIGHT)
    assert seed == 42
    assert prompt == "Hello, world!"


# mypy typing
class MakeMessageProtocol(Protocol):
    def __call__(
        self,
        text: str | None = "hello",
        photo: bool = False,
        reply: bool = False,
        caption: str | None = None,
        reply_to_text: str | None = None,
        reply_to_photo: bool = False,
        reply_to_caption: str | None = None,
    ) -> Mock:
        ...


MakeTgPhotoProtocol = Callable[[], list[Mock]]


@pytest.fixture
def make_tg_photo() -> Callable[[], list[Mock]]:
    def _make_tg_photo() -> list[Mock]:
        photosize = Mock(spec=telegram.PhotoSize)
        tg_file = Mock(spec=telegram.File)
        photosize.get_file.return_value = tg_file
        tg_file.download_as_bytearray.return_value = _PHOTO_BYTES
        return [photosize]

    return _make_tg_photo


@pytest.fixture
def make_message(make_tg_photo: MakeTgPhotoProtocol) -> Callable[[], MakeMessageProtocol]:
    def _make_message(
        text: str | None = "hello",
        photo: bool = False,
        reply: bool = False,
        caption: str | None = None,
        reply_to_text: str | None = None,
        reply_to_photo: bool = False,
        reply_to_caption: str | None = None,
    ) -> Mock:
        message = Mock(spec=telegram.Message)
        message.text = text
        if not photo:
            message.photo = []
        else:
            message.photo = make_tg_photo()
        if reply:
            message.reply_to_message = _make_message(text=reply_to_text, photo=reply_to_photo, caption=reply_to_caption)
        else:
            message.reply_to_message = None
            # Some combinations make no sense
            assert reply_to_text is None
            assert reply_to_photo is False
            assert reply_to_caption is None

        message.caption = caption
        return message

    return _make_message


class MakeUpdateProtocol(Protocol):
    def __call__(self, message: Mock | None = None, query: str | None = None) -> Mock:
        ...


@pytest.fixture
def make_update(make_message: MakeMessageProtocol) -> MakeUpdateProtocol:
    def _make_update(message: Mock | None = None, query: str | None = None) -> Mock:
        update = Mock(spec=telegram.Update)
        update.effective_user.id = 0
        update.effective_chat.id = env.CHAT_ID

        if message is None:
            message = make_message()

        if query is None:
            update.message = message
            update.callback_query = None
        else:
            update.callback_query = Mock(spec=telegram.CallbackQuery)
            update.callback_query.message = message
            update.callback_query.data = query
        return update

    return _make_update


@pytest.fixture
def update(make_update: MakeUpdateProtocol) -> Mock:
    return make_update()


# TODO: Move these perms test to the bot tests
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


def assert_called_once_without_image(provider: ModelProvider) -> None:
    fill_default_args(provider)  # type: ignore[arg-type]  # FIXME
    assert_called_once_with_partial(provider, init_image=None)  # type: ignore[arg-type]


def assert_called_once_with_image(provider: ModelProvider) -> None:
    provider.assert_called_once()  # type: ignore[attr-defined]
    fill_default_args(provider)  # type: ignore[arg-type]  # FIXME
    assert isinstance(provider.call_args.kwargs["init_image"], np.ndarray)  # type: ignore[attr-defined]


@pytest.fixture(params=[False, True])
def tryagain(request: FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


# FIXME move some stuff to test_local_model_provider.py
@pytest.fixture(params=["update", "noupdate"])
def bot_(mocker: MockerFixture, request: FixtureRequest, model_provider_: Mock) -> bot.Bot:
    if request.param == "noupdate":
        mocker.patch("local_model_provider.UPDATE_MODEL", False)
    else:
        assert request.param == "update"
        mocker.patch("local_model_provider.UPDATE_MODEL", True)
    b = bot.Bot(model_provider_)
    return b


def test_construct_bot(bot_: bot.Bot) -> None:
    pass


MakeContextProtocol = Callable[[], Mock]


@pytest.fixture
def make_context() -> MakeContextProtocol:
    def _make_context() -> Mock:
        context = Mock(spec=telegram.ext.ContextTypes.DEFAULT_TYPE)
        context.bot = Mock(spec=telegram.Bot)
        return context

    return _make_context


@dataclass
class BotEnv:
    bot: bot.Bot
    update: Mock
    context: Mock
    message: Mock


class MakeBotEnvProtocol(Protocol):
    def __call__(self, message: Mock, query: str | None = None) -> BotEnv:
        ...


@pytest.fixture
def make_botenv(
    bot_: bot.Bot, make_update: MakeUpdateProtocol, make_context: MakeContextProtocol
) -> MakeBotEnvProtocol:
    def _make_botenv(message: Mock, query: str | None = None) -> BotEnv:
        return BotEnv(
            bot=bot_, update=make_update(message=message, query=query), context=make_context(), message=message
        )

    return _make_botenv


async def test_bot_no_command_no_photo(make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol) -> None:
    message = make_message(text="hello")
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_not_called()
    botenv.bot.model_provider.assert_not_called()  # type: ignore[attr-defined]


async def test_bot_command(make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol) -> None:
    message = make_message(text=bot.COMMAND + "hello")
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_called_once()
    logger.info("kwargs={}", botenv.bot.model_provider.call_args.kwargs)  # type: ignore[attr-defined]
    fill_default_args(botenv.bot.model_provider)
    logger.info("kwargs={}", botenv.bot.model_provider.call_args.kwargs)  # type: ignore[attr-defined]
    assert_called_once_without_image(botenv.bot.model_provider)
    assert botenv.bot.model_provider.call_args.kwargs["prompt"] == "hello"  # type: ignore[attr-defined]


async def test_bot_photo_without_caption(make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol) -> None:
    message = make_message(photo=True)
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_not_called()
    message.photo[-1].get_file.assert_not_called()
    botenv.bot.model_provider.assert_not_called()  # type: ignore[attr-defined]


async def test_bot_photo_with_caption(make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol) -> None:
    message = make_message(photo=True, caption="hello")
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_called_once()
    message.photo[-1].get_file.assert_called()
    botenv.bot.model_provider.assert_called_once()  # type: ignore[attr-defined]
    assert_called_once_with_image(botenv.bot.model_provider)
    assert botenv.bot.model_provider.call_args.kwargs["prompt"] == "hello"  # type: ignore[attr-defined]


async def test_bot_reply_to_photo_no_command(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    message = make_message(reply=True, reply_to_photo=True)
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_not_called()
    message.reply_to_message.photo[-1].get_file.assert_not_called()
    botenv.bot.model_provider.assert_not_called()  # type: ignore[attr-defined]


async def test_bot_reply_to_photo_with_command(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    message = make_message(text=bot.COMMAND + "hello", reply=True, reply_to_photo=True)
    botenv = make_botenv(message)
    await botenv.bot.handle_update(botenv.update, botenv.context, message, tryagain=False)
    message.reply_text.assert_called_once()
    message.reply_to_message.photo[-1].get_file.assert_called_once()
    assert_called_once_with_image(botenv.bot.model_provider)
    assert botenv.bot.model_provider.call_args.kwargs["prompt"] == "hello"  # type: ignore[attr-defined]


async def test_handle_button_try_again_txt2img(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    # 'message' is a mocked txt2img response by the bot; it is a response to a prompt with a command
    message = make_message(
        text=None,
        photo=True,
        reply=True,
        caption='"hello, world" (Seed: 1234)',
        reply_to_text=bot.COMMAND + "hello, world",
    )

    botenv = make_botenv(message, query="TRYAGAIN")
    await botenv.bot.handle_button(botenv.update, botenv.context)
    message.reply_to_message.reply_text.assert_called_once()
    assert_called_once_without_image(botenv.bot.model_provider)
    assert botenv.bot.model_provider.call_args.kwargs["prompt"] == "hello, world"  # type: ignore[attr-defined]


async def test_handle_button_try_again_img2img(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    # 'message' is a mocked img2img response by the bot
    message = make_message(
        text=None,
        photo=True,
        reply_to_photo=True,
        reply=True,
        caption='"hello, world" (Seed: 1234)',
        reply_to_caption='"hello, world" (Seed: 42)',
    )

    botenv = make_botenv(message, query="TRYAGAIN")
    await botenv.bot.handle_button(botenv.update, botenv.context)
    message.reply_to_message.reply_text.assert_called_once()
    assert_called_once_with_image(botenv.bot.model_provider)
    assert botenv.bot.model_provider.call_args.kwargs["prompt"] == "hello, world"  # type: ignore[attr-defined]


async def test_handle_button_try_again_txt2img_seed(
    mocker: MockerFixture, make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    """Test that the seed is re-randomized when the user presses the 'Try Again' button even if the
    orignal prompt had a seed."""
    message = make_message(
        text=None,
        photo=True,
        reply=True,
        caption='"hello, world" (Seed: 1234)',
        reply_to_text=bot.COMMAND + "seed:1234 hello, world",
    )

    botenv = make_botenv(message, query="TRYAGAIN")
    mocker.patch("random.randint", return_value=42)
    mocker.patch("torch.Generator")

    await botenv.bot.handle_button(botenv.update, botenv.context)
    # The reply should be to the original prompt, not the bot's response
    message.reply_to_message.reply_text.assert_called_once()
    assert_called_once_with_partial(botenv.bot.model_provider, prompt="hello, world", seed=42, init_image=None)


async def test_handle_button_variations_txt2img(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    message = make_message(
        text=None,
        photo=True,
        reply=True,
        caption='"hello, world" (Seed: 1234)',
        reply_to_text=bot.COMMAND + "hello, world",
    )

    botenv = make_botenv(message, query="VARIATIONS")
    await botenv.bot.handle_button(botenv.update, botenv.context)
    # The reply should be to the message whose button was clicked
    message.reply_text.assert_called_once()
    assert_called_once_with_partial(botenv.bot.model_provider, prompt="hello, world")
    assert isinstance(botenv.bot.model_provider.call_args.kwargs.get("init_image"), np.ndarray)  # type: ignore[attr-defined]


async def test_handle_button_variations_img2img(
    make_message: MakeMessageProtocol, make_botenv: MakeBotEnvProtocol
) -> None:
    message = make_message(
        text=None,
        photo=True,
        reply_to_photo=True,
        reply=True,
        caption='"hello, world" (Seed: 1234)',
        reply_to_caption='"hello, world" (Seed: 42)',
    )

    botenv = make_botenv(message, query="VARIATIONS")
    await botenv.bot.handle_button(botenv.update, botenv.context)
    # The reply should be to the message whose button was clicked
    message.reply_text.assert_called_once()
    fill_default_args(botenv.bot.model_provider)
    assert_called_once_with_partial(botenv.bot.model_provider, prompt="hello, world")
    assert isinstance(botenv.bot.model_provider.call_args.kwargs["init_image"], np.ndarray)  # type: ignore[attr-defined]
