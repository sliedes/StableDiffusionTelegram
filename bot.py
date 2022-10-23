import asyncio
import random
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
import PIL
import telegram
import telegram.ext
from PIL import Image

import env
from image_to_image import preprocess
from local_model_provider import LocalModelProvider
from model_provider import ModelProvider
from my_logging import logger

# Interpret messages starting with this string as requests to us
COMMAND = "!kuva "

REPORT_TORCH_DUPLICATES = False

gpu_lock = asyncio.Lock()


def image_to_bytes(image: Any) -> bytes:
    bio = BytesIO()
    bio.name = "image.jpeg"
    image.save(bio, "JPEG")
    bio.seek(0)
    return bio.read()


def get_try_again_markup(tryagain: bool = True) -> telegram.InlineKeyboardMarkup:
    if tryagain:
        keyboard = [
            [
                telegram.InlineKeyboardButton("Try again", callback_data="TRYAGAIN"),
            ]
        ]
    else:
        keyboard = [[]]
    keyboard[0].append(telegram.InlineKeyboardButton("Variations", callback_data="VARIATIONS"))

    reply_markup = telegram.InlineKeyboardMarkup(keyboard)
    return reply_markup


def parse_seed(prompt: str) -> tuple[int | None, str]:
    seed: int | None = None
    if prompt.startswith("seed:") and " " in prompt:
        seed_str, prompt = prompt.split(" ", 1)
        seed_str = seed_str.removeprefix("seed:")
        try:
            seed = int(seed_str)
        except ValueError:
            pass
    return seed, prompt


def numpy_to_pil(image: npt.NDArray[np.float32]) -> PIL.Image:
    """
    Convert a numpy image to a PIL image.
    """
    assert image.ndim == 3, image.shape
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


async def generate_image(
    model_provider: ModelProvider,
    prompt: str,
    seed: int | None = None,
    height: int = env.HEIGHT,
    width: int = env.WIDTH,
    num_inference_steps: int = env.NUM_INFERENCE_STEPS,
    strength: float = env.STRENGTH,
    guidance_scale: float = env.GUIDANCE_SCALE,
    photo: bytes | None = None,
    ignore_seed: bool = False,
) -> tuple[PIL.Image, int, str]:
    logger.info("generate_image (photo={}): {}", photo is not None, prompt)
    seed, prompt = parse_seed(prompt)
    if seed is None or ignore_seed:
        seed = random.randint(1, 100000)

    if photo is not None:
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((height, width))
        init_image = preprocess(init_image)
        image = (
            await model_provider(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                init_image=init_image,
            )
        )[0]
    else:
        image = (
            await model_provider(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        )[0]
    return numpy_to_pil(image), seed, prompt


def perms_ok(update: telegram.Update) -> bool:
    assert update.effective_user is not None
    assert update.effective_chat is not None
    if update.effective_user.id != env.ADMIN_ID and update.effective_chat.id != env.CHAT_ID:
        logger.warning("Denied: user_id={}, chat_id={}", update.effective_user.id, update.effective_chat.id)
        return False
    if update.effective_user.id != env.ADMIN_ID and env.ADMIN_ONLY:
        logger.warning(
            "Denied because of admin_only: user_id={}, chat_id={}", update.effective_user.id, update.effective_chat.id
        )
        return False
    return True


@dataclass
class ParsedRequest:
    prompt: str
    photo: bytearray | None
    tryagain: bool  # Whether we should have a "Try Again" button.


QUERY_SEED_RE = re.compile(r'"(.*)" \(Seed: \d+\)')


def extract_query_from_string(prompt: str) -> str:
    logger.debug("extract_query_from_string: {}", prompt)
    while prompt.startswith(COMMAND):
        prompt = prompt.removeprefix(COMMAND)

    while True:
        match = QUERY_SEED_RE.match(prompt)
        if not match:
            break
        prompt = match.group(1)

    logger.debug("extract_query_from_string returning: {}", prompt)

    return prompt


async def parse_request(
    update: telegram.Update, message: telegram.Message, tryagain: bool = True
) -> ParsedRequest | None:
    if not perms_ok(update):
        return None

    logger.debug("parse_request: {}", message)

    # If not None, take photo from this and use it as img2img source
    msg_of_photo: telegram.Message | None = None

    # Use this as the prompt text. Any COMMAND prefix will be removed.
    prompt = ""

    # If it is a photo with a caption, treat it as an img2img request
    if message.photo:
        # For that to work, it needs to have a caption. Otherwise we just ignore it.
        if message.caption is None:
            logger.debug("Message is a photo without caption. Ignoring.")
            return None
        logger.debug("Message has a photo and caption; interpreting as img2img.")
        prompt = message.caption
        msg_of_photo = message
    else:
        # It is not a photo. If it does not start with COMMAND, it's not for us.
        assert message.text is not None
        prompt = message.text
        if not prompt.startswith(COMMAND):
            logger.debug("Not for us: {}", prompt)
            return None

        # If it is not a reply, it's txt2img. If it is a reply, see if we can find a photo.
        if message.reply_to_message and message.reply_to_message.photo:
            logger.debug("Replied to a message with photo; using that photo")
            msg_of_photo = message.reply_to_message
            # If this is a reply to a photo, we will not be able to follow deep enough to find the photo
            # (Telegram bot limitation). Thus, disable the Try Again button.
            tryagain = False

    photo = None
    if msg_of_photo is not None:
        photo = await (await msg_of_photo.photo[-1].get_file()).download_as_bytearray()

    return ParsedRequest(prompt=extract_query_from_string(prompt), photo=photo, tryagain=tryagain)


class Bot:
    model_provider: ModelProvider

    def __init__(self, model_provider: ModelProvider | None = None) -> None:
        if model_provider is None:
            model_provider = LocalModelProvider()
        self.model_provider = model_provider

    async def handle_update(
        self,
        update: telegram.Update,
        context: telegram.ext.ContextTypes.DEFAULT_TYPE,
        message: telegram.Message | None = None,
        tryagain: bool = True,
        ignore_seed: bool = False,
    ) -> None:
        if message is None:
            message = update.message

        assert message is not None
        req = await parse_request(update, message, tryagain)
        if req is None:
            return

        progress_msg = await message.reply_text("Generating image...", reply_to_message_id=message.message_id)
        im, seed, prompt = await generate_image(
            model_provider=self.model_provider, prompt=req.prompt, photo=req.photo, ignore_seed=ignore_seed
        )
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(
            message.chat_id,
            image_to_bytes(im),
            caption=f'"{prompt}" (Seed: {seed})',
            reply_markup=get_try_again_markup(tryagain=req.tryagain),
            reply_to_message_id=message.message_id,
        )

    async def handle_button(
        self, update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE, ignore_seed: bool = False
    ) -> None:
        query = update.callback_query
        assert query is not None
        assert query.message is not None
        assert query.message.reply_to_message is not None
        parent_message = query.message.reply_to_message

        assert update.effective_chat is not None

        await query.answer()

        photo: bytearray | None = None

        if query.data == "TRYAGAIN":
            # for Try Again, reply to the original prompt. We just re-handle the replied to message.
            logger.info("Try again: Re-handling parent.")
            await self.handle_update(update, context, message=parent_message, ignore_seed=True)
            return
        elif query.data != "VARIATIONS":
            logger.error("Unknown query data: {}", query)
            return

        # for Variations, reply to the message whose button is clicked
        reply_to = query.message.message_id
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = parent_message.text if parent_message.text is not None else parent_message.caption
        assert prompt is not None
        prompt = extract_query_from_string(prompt)
        logger.info("Variations: {}", prompt)

        progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=reply_to)
        im, seed, prompt = await generate_image(
            model_provider=self.model_provider, prompt=prompt, photo=photo, ignore_seed=True
        )
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(
            update.effective_chat.id,
            image_to_bytes(im),
            caption=f'"{prompt}" (Seed: {seed})',
            reply_markup=get_try_again_markup(),
            reply_to_message_id=reply_to,
        )


def main() -> None:
    bot = Bot(LocalModelProvider())

    if REPORT_TORCH_DUPLICATES:
        import torch_duplicates

        torch_duplicates.report_dups_in_memory(logger)

    app = telegram.ext.ApplicationBuilder().token(env.TG_TOKEN).build()

    app.add_handler(
        telegram.ext.MessageHandler(
            ~telegram.ext.filters.UpdateType.EDITED_MESSAGE
            & ((telegram.ext.filters.TEXT & ~telegram.ext.filters.COMMAND) | telegram.ext.filters.PHOTO),
            bot.handle_update,
            block=False,
        )
    )
    app.add_handler(telegram.ext.MessageHandler(telegram.ext.filters.PHOTO, bot.handle_update, block=False))
    app.add_handler(telegram.ext.CallbackQueryHandler(bot.handle_button, block=False))

    logger.info("Starting.")
    app.run_polling(read_timeout=20)


if __name__ == "__main__":
    main()
