model_server_pb2.py model_server_pb2_grpc.py: model_server.proto
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model_server.proto
