import asyncio
from dataclasses import dataclass, field
import threading

# from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Any
from io import BytesIO

from my_logging import logger

from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess

import grpc

import env
import config

import model_server_pb2_grpc
from model_server_pb2 import Metadata, Request, Response

import torch
from torch import autocast
from PIL import Image

DEFAULT_PRIORITY = 100

# disable safety checker if wanted
def _dummy_checker(images, **kwargs):
    return images, False


# Wrapper for requests to make them compatible with PriorityQueue
@dataclass(order=True)
class RequestQueueItem:
    request: Request = field(compare=False)
    priority: int = field(default=DEFAULT_PRIORITY)


class GPUWorker:
    _pipe: StableDiffusionPipeline
    _img2imgPipe: StableDiffusionImg2ImgPipeline
    _request_queue: asyncio.PriorityQueue[RequestQueueItem]
    _eventloop: asyncio.AbstractEventLoop
    _thread: threading.Thread

    def __init__(self):
        self.gpu_lock = asyncio.Lock()
        self._request_queue: asyncio.PriorityQueue[RequestQueueItem] = asyncio.PriorityQueue()
        self._eventloop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._worker_main, name="GPUWorker")

    def _worker_main(self):
        logger.info("Loading text2img pipeline")
        self._pipe = StableDiffusionPipeline.from_pretrained(
            env.MODEL_DATA, revision=env.MODEL_REVISION, torch_dtype=env.TORCH_DTYPE, use_auth_token=env.HF_AUTH_TOKEN
        )
        logger.info("Loaded text2img pipeline")
        self._pipe = self._pipe.to("cpu")

        logger.info("Loading img2img pipeline")
        self._img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            env.MODEL_DATA, revision=env.MODEL_REVISION, torch_dtype=env.TORCH_DTYPE, use_auth_token=env.HF_AUTH_TOKEN
        )
        logger.info("Loaded img2img pipeline")
        self._img2imgPipe = self._img2imgPipe.to("cpu")

        if not env.SAFETY_CHECKER:
            self._pipe.safety_checker = _dummy_checker
            self._img2imgPipe.safety_checker = _dummy_checker

        while True:
            rqitem = asyncio.run_coroutine_threadsafe(self._request_queue.get(), self._eventloop).result()
            logger.debug("Worker received request with priority {}", rqitem.priority)

    def _generate_image(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_inference_steps: int,
        strength: float,
        guidance_scale: float,
        photo: Optional[bytes] = None,
    ) -> Tuple[Image.Image, str]:
        logger.info("generate_image (photo={}): {}", photo is not None, prompt)
        generator = torch.cuda.manual_seed_all(seed)

        if photo is not None:
            self._pipe.to("cpu")
            self._img2imgPipe.to("cuda")
            init_image = Image.open(BytesIO(photo)).convert("RGB")
            init_image = init_image.resize((height, width))
            init_image = preprocess(init_image)
            with autocast("cuda"):
                self.image = self._img2imgPipe(
                    prompt=[prompt],
                    init_image=init_image,
                    generator=generator,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]
        else:
            self._pipe.to("cuda")
            self._img2imgPipe.to("cpu")
            with autocast("cuda"):
                image = self._pipe(
                    prompt=[prompt],
                    generator=generator,
                    strength=strength,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]
        return image

    # Call from main thread
    async def generate_image(self, request: Request) -> Response:
        async with self.gpu_lock():
            image = await asyncio.to_thread(
                self._generate_image,
                kwargs=dict(
                    prompt=request.prompt,
                    seed=request.seed,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.iterations,
                    strength=request.strength,
                    guidance_scale=request.guidance_scale,
                    photo=request.image,
                ),
            )


class ModelServicer(model_server_pb2_grpc.ImGenServiceServicer):
    async def generate_image(self, request: Request, context: grpc.ServicerContext) -> Response:
        response = Response()
        response.prompt = "test"
        response.iterations = 100
        return response


async def serve():
    server = grpc.aio.server()
    model_server_pb2_grpc.add_ImGenServiceServicer_to_server(ModelServicer(), server)
    endpoint = f"{config.SERVER_LISTEN_ADDR}:{config.SERVER_PORT}"
    logger.info("Listening to {}", endpoint)
    server.add_insecure_port(endpoint)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve(), debug=True)
