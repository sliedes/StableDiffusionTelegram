#!/usr/bin/env python3

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
from model_server_pb2 import (
    ImGenResponseMetadata,
    ImGenRequest,
    ImGenResponse,
    TokenizeResponse,
    TokenizeRequest,
)
from google.protobuf.timestamp_pb2 import Timestamp

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
    request: ImGenRequest = field(compare=False)
    response: ImGenResponse = field(compare=False)  # some fields prepopulated when inserted
    future: Optional[asyncio.Future[ImGenResponse]] = field(compare=False)
    priority: int = field(default=DEFAULT_PRIORITY)


# TODO: more graceful handling of KeyboardInterrupt / a stop() method etc.

# Anything named _a_* must only be used from the executor thread ("async context").
# Anything named _w_* must only be used from the GPU worker thread.
class GPUWorker:
    _a_pipe: StableDiffusionPipeline
    _a_img2imgpipe: StableDiffusionImg2ImgPipeline
    _a_request_queue: asyncio.PriorityQueue[RequestQueueItem]
    _eventloop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _start_worker_task: asyncio.Task

    def __init__(self):
        self._gpu_lock = asyncio.Lock()
        self._a_request_queue: asyncio.PriorityQueue[RequestQueueItem] = asyncio.PriorityQueue()
        self._eventloop = asyncio.get_event_loop()
        self._start_worker_task = asyncio.create_task(self._start_worker(), name="_start_worker")

    @logger.catch
    async def _start_worker(self) -> None:
        logger.info("Loading text2img pipeline")
        StableDiffusionPipeline.from_pretrained = logger.catch(StableDiffusionPipeline.from_pretrained)
        a_pipe_fut = asyncio.to_thread(
            StableDiffusionPipeline.from_pretrained,
            pretrained_model_name_or_path=env.MODEL_DATA,
            revision=env.MODEL_REVISION,
            torch_dtype=env.TORCH_DTYPE,
            use_auth_token=env.HF_AUTH_TOKEN,
        )

        logger.info("Loading img2img pipeline")
        a_img2imgpipe_fut = asyncio.to_thread(
            StableDiffusionImg2ImgPipeline.from_pretrained,
            pretrained_model_name_or_path=env.MODEL_DATA,
            revision=env.MODEL_REVISION,
            torch_dtype=env.TORCH_DTYPE,
            use_auth_token=env.HF_AUTH_TOKEN,
        )

        self._a_pipe = (await a_pipe_fut).to("cpu")
        logger.info("Loaded text2img pipeline")
        self._a_img2imgpipe = (await a_img2imgpipe_fut).to("cpu")
        logger.info("Loaded img2img pipeline")

        if not env.SAFETY_CHECKER:
            self._a_pipe.safety_checker = _dummy_checker
            self._a_img2imgpipe.safety_checker = _dummy_checker

        self._thread = threading.Thread(target=self._w_worker_main, name="GPUWorker")

    @logger.catch
    def _w_worker_main(self) -> None:
        while True:
            rqitem = asyncio.run_coroutine_threadsafe(self._a_request_queue.get(), self._eventloop).result()
            logger.debug("Worker received request with priority {}", rqitem.priority)

    @logger.catch
    def _w_generate_image(
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
            self._a_pipe.to("cpu")
            self._a_img2imgpipe.to("cuda")
            init_image = Image.open(BytesIO(photo)).convert("RGB")
            init_image = init_image.resize((height, width))
            init_image = preprocess(init_image)
            with autocast("cuda"):
                self.image = self._a_img2imgpipe(
                    prompt=[prompt],
                    init_image=init_image,
                    generator=generator,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]
        else:
            self._a_pipe.to("cuda")
            self._a_img2imgpipe.to("cpu")
            with autocast("cuda"):
                image = self._a_pipe(
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
    @logger.catch
    async def generate_image_now(self, request: ImGenRequest) -> ImGenResponse:
        async with self._gpu_lock():
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

    @logger.catch
    async def _generate_image_via_queue(self, request: ImGenRequest) -> ImGenResponse:
        response = ImGenResponse()
        response.req_metadata = request.req_metadata
        response.resp_metadata = ImGenResponseMetadata()
        response.resp_metadata.request_received_time = Timestamp.GetCurrentTime()
        item = RequestQueueItem(request=request, response=response, priority=request.priority)
        item.resp_metadata.queue_size = self._a_request_queue.qsize()
        item.future = self._eventloop.create_future()
        self._a_request_queue.put_nowait(item)
        return await item.future

    @logger.catch
    async def generate_image(self, request: ImGenRequest) -> ImGenResponse:
        logger.debug("generate_image request")
        return await self._generate_image_via_queue(request)

    @logger.catch
    async def tokenize_prompt(self, request: TokenizeRequest) -> TokenizeResponse:
        logger.debug("tokenize_prompt: {}", request.prompt)
        await self._start_worker_task
        return TokenizeResponse(prompt_tokens=self._a_pipe.tokenizer.tokenize(request.prompt))


class ModelServicer(model_server_pb2_grpc.ImGenServiceServicer):
    def __init__(self, worker: GPUWorker):
        self.worker = worker

    @logger.catch
    async def generate_image(self, request: ImGenRequest, context: grpc.aio.ServicerContext) -> ImGenResponse:
        return ImGenResponse(prompt="test", iterations=100)

    @logger.catch
    async def tokenize_prompt(self, request: TokenizeRequest, context: grpc.aio.ServicerContext) -> TokenizeResponse:
        return await self.worker.tokenize_prompt(request)


async def serve():
    server = grpc.aio.server()
    model_server_pb2_grpc.add_ImGenServiceServicer_to_server(ModelServicer(GPUWorker()), server)
    endpoint = f"{config.SERVER_LISTEN_ADDR}:{config.SERVER_PORT}"
    logger.info("Listening to {}", endpoint)
    server.add_insecure_port(endpoint)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve(), debug=True)
