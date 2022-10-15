TGTS=model_server_pb2.py model_server_pb2.pyi model_server_pb2_grpc.py model_server_pb2_grpc.pyi

all: $(TGTS)

$(TGTS): model_server.proto
	python -m grpc_tools.protoc -I. --python_out=. --mypy_out=. --grpc_python_out=. --mypy_grpc_out=. model_server.proto
