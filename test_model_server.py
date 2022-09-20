#!/usr/bin/env python3

from typing import Any, Dict

import pytest
import grpc

from model_server_pb2 import (
    ImGenRequest,
    ImGenRequestMetadata,
    ImGenResponse,
    ImGenResponseMetadata,
    TokenizeRequest,
    TokenizeResponse,
)
from model_server_pb2_grpc import ImGenServiceStub

import config
import env

_ImGenRequestMetadata_template = ImGenRequestMetadata(
    width=env.WIDTH,
    height=env.HEIGHT,
    seed=1234,
    iterations=env.NUM_INFERENCE_STEPS,
    strength=env.STRENGTH,
    guidance_scale=env.GUIDANCE_SCALE,
)


def make_ImGenRequest(**kwargs: Dict[str, Any]) -> ImGenRequest:
    meta = ImGenRequestMetadata()
    meta.CopyFrom(_ImGenRequestMetadata_template)
    for k, v in kwargs.items():
        setattr(meta, k, v)
    return ImGenRequest(req_metadata=meta)


def main():
    channel = grpc.insecure_channel(f"localhost:{config.SERVER_PORT}")
    imgen = ImGenServiceStub(channel)

    print(imgen.tokenize_prompt(TokenizeRequest(prompt="Hello, world!")).prompt_tokens)
    resp = imgen.generate_image(make_ImGenRequest(prompt="A ball on a floor", test_no_compute=False, iterations=2))
    resp.image = f"{len(resp.image)} bytes of data".encode("utf-8")
    print(resp)


if __name__ == "__main__":
    main()
