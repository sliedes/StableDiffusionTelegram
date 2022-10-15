#!/usr/bin/env python3

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt
import PIL
import torch
from diffusers import StableDiffusionPipeline
from google.protobuf.timestamp_pb2 import Timestamp
from torch import autocast  # type: ignore[attr-defined]

import config
import env
import model_server_pb2_grpc
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from model_server_pb2 import (
    ImGenRequest,
    ImGenResponse,
    ImGenResponseMetadata,
    TokenizeRequest,
    TokenizeResponse,
)
from my_logging import logger


# disable safety checker if wanted
def _dummy_checker(images: Any, **kwargs: Any) -> tuple[Any, bool]:
    return images, False


# Wrapper for requests to make them compatible with PriorityQueue
@dataclass(order=True)
class RequestQueueItem:
    request: ImGenRequest | None = field(compare=False)  # None requests stopping worker
    response: ImGenResponse = field(compare=False)  # some fields prepopulated when inserted
    future: asyncio.Future[ImGenResponse] | None = field(compare=False)
    priority: int


def timestamp_now() -> Timestamp:
    t = Timestamp()
    t.GetCurrentTime()
    return t


# TODO: more graceful handling of KeyboardInterrupt / a stop() method etc.

# Anything named _a_* must only be used from the executor thread ("async context").
# Anything named _w_* must only be used from the GPU worker thread.
class GPUWorker:
    _a_pipe: StableDiffusionPipeline
    _a_img2imgpipe: StableDiffusionImg2ImgPipeline
    _a_request_queue: asyncio.PriorityQueue[RequestQueueItem]
    _eventloop: asyncio.AbstractEventLoop
    _worker_thread: threading.Thread
    _start_worker_task: asyncio.Task[None]

    def _assert_async(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("Method was not called from inside event loop")

    def _assert_worker(self) -> None:
        assert threading.get_ident() == self._worker_thread.ident, "Method was not called from worker thread"

    def __init__(self) -> None:
        self._a_request_queue: asyncio.PriorityQueue[RequestQueueItem] = asyncio.PriorityQueue()
        self._eventloop = asyncio.get_event_loop()
        self._start_worker_task = asyncio.create_task(self._start_worker(), name="_start_worker")

    @logger.catch
    async def stop(self, stop_prio: int = 0) -> None:
        self._assert_async()
        logger.debug("stop called")
        sentinel = RequestQueueItem(request=None, response=ImGenResponse(), future=None, priority=stop_prio)
        self._a_request_queue.put_nowait(sentinel)
        await asyncio.to_thread(self._worker_thread.join)  # spawning a thread to join a thread feels wrong...

    def _load_txt2img(self) -> StableDiffusionPipeline:
        return StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=env.MODEL_DATA,
            revision=env.MODEL_REVISION,
            torch_dtype=env.TORCH_DTYPE,
            use_auth_token=env.HF_AUTH_TOKEN,
        ).to("cuda")

    @logger.catch
    async def _start_worker(self) -> None:
        self._assert_async()
        logger.info("Loading txt2img pipeline")
        self._a_pipe = await asyncio.to_thread(self._load_txt2img)
        logger.info("Loaded txt2img pipeline")

        self._a_img2imgpipe = StableDiffusionImg2ImgPipeline(
            vae=self._a_pipe.vae,
            text_encoder=self._a_pipe.text_encoder,
            tokenizer=self._a_pipe.tokenizer,
            unet=self._a_pipe.unet,
            scheduler=self._a_pipe.scheduler,
            safety_checker=self._a_pipe.safety_checker,
            feature_extractor=self._a_pipe.feature_extractor,
        ).to("cuda")
        logger.info("Constructed img2img pipeline")

        if not env.SAFETY_CHECKER:
            self._a_pipe.safety_checker = _dummy_checker
            self._a_img2imgpipe.safety_checker = _dummy_checker

        self._worker_thread = threading.Thread(target=self._w_worker_main, name="GPUWorker")
        self._worker_thread.start()

    @logger.catch
    def _w_worker_main(self) -> None:
        self._assert_worker()
        logger.info("_w_worker_main started")
        while True:
            try:
                rqitem = asyncio.run_coroutine_threadsafe(self._a_request_queue.get(), self._eventloop).result()
                logger.debug("Worker received request with priority {}", rqitem.priority)
                req = rqitem.request
                if req is None:
                    # Sentinel, quit
                    logger.debug("Worker received sentinel, quitting.")
                    return
                meta = req.req_metadata
                resp = rqitem.response
                resp.resp_metadata.start_processing_time.GetCurrentTime()
                if meta.test_no_compute:
                    im: np.ndarray[Any, np.dtype[np.float16]] = np.ndarray((0, 0), dtype=np.float16)
                    prompt = ""
                else:
                    photo = b""
                    if req.HasField("image"):
                        photo = req.image.data
                    im, prompt = self._w_generate_image(
                        prompt=meta.prompt,
                        seed=meta.seed,
                        height=meta.height,
                        width=meta.width,
                        num_inference_steps=meta.iterations,
                        strength=meta.strength,
                        guidance_scale=meta.guidance_scale,
                        photo=photo,
                    )
                resp.resp_metadata.finish_processing_time.GetCurrentTime()
                resp.image.data = im.tobytes()
                resp.image.width = im.shape[0]  # TODO: check that width and height are this way...
                resp.image.height = im.shape[1]
                resp.image.dtype = "B"
                logger.debug(
                    "_w_worker_main: Got image of {} bytes, shape={}, dtype={}",
                    len(resp.image.data),
                    im.shape,
                    im.dtype,
                )
                # TODO: prompt_tokens
                assert rqitem.future is not None
                self._eventloop.call_soon_threadsafe(rqitem.future.set_result, resp)
            except Exception as e:
                assert rqitem.future is not None
                self._eventloop.call_soon_threadsafe(rqitem.future.set_exception, e)

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
        photo: bytes = b"",
    ) -> tuple[npt.NDArray[np.float_], str]:
        self._assert_worker()
        logger.info("generate_image (photo={}): {}", bool(photo), prompt)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        if photo:
            # TODO: float buffer
            init_image = PIL.Image.fromarray(np.frombuffer(photo, dtype="B").reshape((width, height, 3)), mode="RGB")
            init_image = init_image.resize((height, width))
            init_image = preprocess(init_image).half()
            with autocast("cuda"):
                image = self._a_img2imgpipe(
                    prompt=[prompt],
                    init_image=init_image,
                    generator=generator,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="numpy",
                )["sample"][0]
        else:
            with autocast("cuda"):
                image = self._a_pipe(
                    prompt=[prompt],
                    generator=generator,
                    strength=strength,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="numpy",
                )["sample"][0]
        return image, ""

    @logger.catch
    async def _generate_image_via_queue(self, request: ImGenRequest) -> ImGenResponse:
        self._assert_async()
        response = ImGenResponse(
            req_metadata=request.req_metadata,
            resp_metadata=ImGenResponseMetadata(
                request_received_time=timestamp_now(), queue_size=self._a_request_queue.qsize()
            ),
        )
        item = RequestQueueItem(
            request=request,
            response=response,
            priority=request.req_metadata.priority,
            future=self._eventloop.create_future(),
        )
        self._a_request_queue.put_nowait(item)
        assert item.future is not None
        return await item.future

    @logger.catch
    async def generate_image(self, request: ImGenRequest) -> ImGenResponse:
        self._assert_async()
        logger.debug("generate_image (img2img={}): {}", request.HasField("image"), request.req_metadata.prompt)
        return await self._generate_image_via_queue(request)

    @logger.catch
    async def tokenize_prompt(self, request: TokenizeRequest) -> TokenizeResponse:
        self._assert_async()
        logger.debug("tokenize_prompt: {}", request.prompt)
        await self._start_worker_task
        return TokenizeResponse(prompt_tokens=self._a_pipe.tokenizer.tokenize(request.prompt))


def _assert_has_field(context: grpc.aio.ServicerContext, msg: Any, field: str) -> None:  # type: ignore[name-defined]
    if not msg.HasField(field):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"field `{field}` is mandatory")


class ModelServicer(model_server_pb2_grpc.ImGenServiceServicer):
    def __init__(self, worker: GPUWorker) -> None:
        self.worker = worker

    @logger.catch
    async def generate_image(  # type: ignore[override]
        self, request: ImGenRequest, context: grpc.aio.ServicerContext  # type: ignore[name-defined]
    ) -> ImGenResponse:
        # validate request
        _assert_has_field(context, request, "req_metadata")
        if request.HasField("image"):
            if request.image.dtype != "B":
                # TODO: handle different dtypes, e.g. convert them here
                context.abort(grpc.StatusCode.UNIMPLEMENTED, "non-RGB8 (B) dtypes are not implemented")
            if request.image.width != env.WIDTH or request.image.height != env.HEIGHT:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"width and height must currently be the configured {env.WIDTH} and {env.HEIGHT}",
                )
            exp_len = request.image.width * request.image.height * 3
            if len(request.image.data) != exp_len:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Expected {exp_len} bytes of image data (w*h*3 bytes), but got {len(request.image.data)} bytes",
                )

        return await self.worker.generate_image(request)

    @logger.catch
    async def tokenize_prompt(self, request: TokenizeRequest, context: grpc.aio.ServicerContext) -> TokenizeResponse:  # type: ignore[name-defined,override]
        return await self.worker.tokenize_prompt(request)


@logger.catch
async def start_server(endpoint: str | None = None) -> grpc.aio.Server:  # type: ignore[name-defined]
    logger.info("start_server: endpoint={}", endpoint)
    server = grpc.aio.server()  # type: ignore[attr-defined]
    model_server_pb2_grpc.add_ImGenServiceServicer_to_server(ModelServicer(GPUWorker()), server)
    endpoint = endpoint or f"{config.SERVER_LISTEN_ADDR}:{config.SERVER_PORT}"
    logger.info("Listening to {}", endpoint)
    server.add_insecure_port(endpoint)
    await server.start()
    return server


@logger.catch
async def serve(endpoint: str | None = None) -> None:
    server = await start_server(endpoint=endpoint)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve(), debug=True)
