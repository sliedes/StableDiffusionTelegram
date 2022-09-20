#!/usr/bin/env python3

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

channel = grpc.insecure_channel(f"localhost:{config.SERVER_PORT}")
imgen = ImGenServiceStub(channel)

for i in range(10):
    print(imgen.tokenize_prompt(TokenizeRequest(prompt="Hello, world!")).prompt_tokens)
