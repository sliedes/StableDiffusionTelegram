import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from PIL import Image

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from io import BytesIO
import random

from typing import Optional, Tuple, Any

from loguru import logger

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
MODEL_DATA = os.getenv("MODEL_DATA", "CompVis/stable-diffusion-v1-4")
LOW_VRAM_MODE = os.getenv("LOW_VRAM", "true").lower() == "true"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", False)
SAFETY_CHECKER = os.getenv("SAFETY_CHECKER", "true").lower() == "true"
HEIGHT = int(os.getenv("HEIGHT", "512"))
WIDTH = int(os.getenv("WIDTH", "512"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "100"))
STRENGTH = float(os.getenv("STRENGTH", "0.75"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
ADMIN_ID = int(os.getenv("ADMIN_ID"))
CHAT_ID = int(os.getenv("CHAT_ID"))

revision = "fp16" if LOW_VRAM_MODE else None
torch_dtype = torch.float16 if LOW_VRAM_MODE else None

# load the text2img pipeline
logger.info("Loading text2img pipeline")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=HF_AUTH_TOKEN
)
pipe = pipe.to("cpu")

# load the img2img pipeline
logger.info("Loading img2img pipeline")
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=HF_AUTH_TOKEN
)
img2imgPipe = img2imgPipe.to("cpu")

# disable safety checker if wanted
def dummy_checker(images, **kwargs):
    return images, False


if not SAFETY_CHECKER:
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker


def image_to_bytes(image) -> BytesIO:
    bio = BytesIO()
    bio.name = "image.jpeg"
    image.save(bio, "JPEG")
    bio.seek(0)
    return bio


def get_try_again_markup() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("Try again", callback_data="TRYAGAIN"),
            InlineKeyboardButton("Variations", callback_data="VARIATIONS"),
        ]
    ]
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


def generate_image(
    prompt: str,
    seed: Optional[int] = None,
    height: int = HEIGHT,
    width: int = WIDTH,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE_SCALE,
    photo: Optional[bytes] = None,
    ignore_seed: bool = False,
) -> Tuple[Any, int, str]:
    logger.info("generate_image: {}", prompt)
    seed, prompt = parse_seed(prompt)
    if seed is None or ignore_seed:
        seed = random.randint(1, 100000)
    generator = torch.cuda.manual_seed_all(seed)

    if photo is not None:
        pipe.to("cpu")
        img2imgPipe.to("cuda")
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
        pipe.to("cuda")
        img2imgPipe.to("cpu")
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


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.debug("generate_and_send_photo")
    if update.effective_user.id != ADMIN_ID and update.message.chat_id != CHAT_ID:
        logger.warning("Denied: user_id={}, chat_id={}", update.effective_user.id, update.message.chat_id)
        return
    prompt = update.message.text
    if not prompt.startswith("!kuva "):
        logger.debug('Not for me: "{}"', prompt)
        return
    prompt = prompt.removeprefix("!kuva ")
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed, prompt = generate_image(prompt=prompt)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(
        update.message.chat_id,
        image_to_bytes(im),
        caption=f'"{prompt}" (Seed: {seed})',
        reply_markup=get_try_again_markup(),
        reply_to_message_id=update.message.message_id,
    )


async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("generate_and_send_from_photo: {}", update.message.caption)
    if update.message.caption is None:
        await update.message.reply_text(
            "The photo must contain a text in the caption", reply_to_message_id=update.message.message_id
        )
        return
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    prompt = update.message.caption.removeprefix("!kuva ")
    im, seed, prompt = generate_image(prompt=prompt, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(
        update.message.chat_id,
        image_to_bytes(im),
        caption=f'"{prompt}" (Seed: {seed})',
        reply_markup=get_try_again_markup(),
        reply_to_message_id=update.message.message_id,
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            prompt = replied_message.caption
            prompt = prompt.removeprefix("!kuva ")
            logger.info("Try again (img2img): {}", prompt)
            im, seed, prompt = generate_image(prompt, photo=photo, ignore_seed=True)
        else:
            prompt = replied_message.text.removeprefix("!kuva ")
            logger.info("Try again: {}", prompt)
            im, seed, prompt = generate_image(prompt, ignore_seed=True)
    elif query.data == "VARIATIONS":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        prompt = prompt.removeprefix("!kuva ")
        im, seed, prompt = generate_image(prompt, photo=photo, ignore_seed=True)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(
        update.effective_chat.id,
        image_to_bytes(im),
        caption=f'"{prompt}" (Seed: {seed})',
        reply_markup=get_try_again_markup(),
        reply_to_message_id=replied_message.message_id,
    )


app = ApplicationBuilder().token(TG_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(MessageHandler(filters.PHOTO, generate_and_send_photo_from_photo))
app.add_handler(CallbackQueryHandler(button))

logger.info("Starting.")

app.run_polling()
