import asyncio
from dataclasses import dataclass
import random
from io import BytesIO
from typing import Optional, Tuple, Any
import re

import torch
from torch import autocast
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from my_logging import logger

from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess

import env

# Interpret messages starting with this string as requests to us
COMMAND = "!kuva "

UPDATE_MODEL = False
REPORT_TORCH_DUPLICATES = True

gpu_lock = asyncio.Lock()


def image_to_bytes(image) -> BytesIO:
    bio = BytesIO()
    bio.name = "image.jpeg"
    image.save(bio, "JPEG")
    bio.seek(0)
    return bio


def get_try_again_markup(tryagain=True) -> InlineKeyboardMarkup:
    if tryagain:
        keyboard = [
            [
                InlineKeyboardButton("Try again", callback_data="TRYAGAIN"),
            ]
        ]
    else:
        keyboard = [[]]
    keyboard[0].append(InlineKeyboardButton("Variations", callback_data="VARIATIONS"))

    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def parse_seed(prompt: str) -> Tuple[Optional[int], str]:
    seed: Optional[int] = None
    if prompt.startswith("seed:") and " " in prompt:
        seed_str, prompt = prompt.split(" ", 1)
        seed_str = seed_str.removeprefix("seed:")
        try:
            seed = int(seed_str)
        except ValueError:
            pass
    return seed, prompt


async def generate_image(
    pipe: StableDiffusionPipeline,
    img2imgPipe: StableDiffusionImg2ImgPipeline,
    prompt: str,
    seed: Optional[int] = None,
    height: int = env.HEIGHT,
    width: int = env.WIDTH,
    num_inference_steps: int = env.NUM_INFERENCE_STEPS,
    strength: float = env.STRENGTH,
    guidance_scale: float = env.GUIDANCE_SCALE,
    photo: Optional[bytes] = None,
    ignore_seed: bool = False,
) -> Tuple[Any, int, str]:
    async with gpu_lock:
        logger.info("generate_image (photo={}): {}", photo is not None, prompt)
        seed, prompt = parse_seed(prompt)
        if seed is None or ignore_seed:
            seed = random.randint(1, 100000)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        if photo is not None:
            init_image = Image.open(BytesIO(photo)).convert("RGB")
            init_image = init_image.resize((height, width))
            init_image = preprocess(init_image)
            with autocast("cuda"):
                image = img2imgPipe(
                    prompt=[prompt],
                    init_image=init_image,
                    generator=generator,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]
        else:
            with autocast("cuda"):
                image = pipe(
                    prompt=[prompt],
                    generator=generator,
                    strength=strength,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]
        return image, seed, prompt


def perms_ok(update: Update) -> bool:
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
    photo: Optional[bytearray]
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


async def parse_request(update: Update, message: Message, tryagain: bool = True) -> Optional[ParsedRequest]:
    if not perms_ok(update):
        return None

    logger.debug("parse_request: {}", message)

    # If not None, take photo from this and use it as img2img source
    msg_of_photo: Optional[Message] = None

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
    pipe: StableDiffusionPipeline
    img2imgPipe: StableDiffusionImg2ImgPipeline

    def __init__(self):
        self.pipe, self.img2imgPipe = load_models()

    async def handle_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        message: Optional[Message] = None,
        tryagain: bool = True,
    ) -> None:
        if message is None:
            message = update.message

        assert message is not None
        req = await parse_request(update, message, tryagain)
        if req is None:
            return

        progress_msg = await message.reply_text("Generating image...", reply_to_message_id=message.message_id)
        im, seed, prompt = await generate_image(
            pipe=self.pipe, img2imgPipe=self.img2imgPipe, prompt=req.prompt, photo=req.photo
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
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        assert query is not None
        assert query.message is not None
        assert query.message.reply_to_message is not None
        parent_message = query.message.reply_to_message

        assert update.effective_chat is not None

        await query.answer()

        photo: Optional[bytearray] = None

        if query.data == "TRYAGAIN":
            # for Try Again, reply to the original prompt. We just re-handle the replied to message.
            # TODO: ignore any seed.
            logger.info("Try again: Re-handling parent.")
            await self.handle_update(update, context, message=parent_message)
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
            pipe=self.pipe, img2imgPipe=self.img2imgPipe, prompt=prompt, photo=photo, ignore_seed=True
        )
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(
            update.effective_chat.id,
            image_to_bytes(im),
            caption=f'"{prompt}" (Seed: {seed})',
            reply_markup=get_try_again_markup(),
            reply_to_message_id=reply_to,
        )


# disable safety checker if wanted
def dummy_checker(images, **kwargs):
    return images, False


def load_models() -> Tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
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
        pipe = torch.load("text2img.pt")
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


def main():
    bot = Bot()

    if REPORT_TORCH_DUPLICATES:
        import torch_duplicates

        torch_duplicates.report_dups_in_memory(logger)

    app = ApplicationBuilder().token(env.TG_TOKEN).build()

    app.add_handler(
        MessageHandler(
            ~filters.UpdateType.EDITED_MESSAGE & ((filters.TEXT & ~filters.COMMAND) | filters.PHOTO), bot.handle_update
        )
    )
    app.add_handler(MessageHandler(filters.PHOTO, bot.handle_update))
    app.add_handler(CallbackQueryHandler(bot.handle_button))

    logger.info("Starting.")
    app.run_polling(read_timeout=20)


if __name__ == "__main__":
    main()
