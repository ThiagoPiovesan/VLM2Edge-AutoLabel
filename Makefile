export UV_PROJECT_ENVIRONMENT ?= /home/piovesant/.venvs/vlm2edge
TACO_LIMIT ?= 500

.PHONY: sync up-redis up-vllm down-vllm up-gguf up-mlflow download-taco enqueue reset-dlq worker eval fiftyone split-yolo train-yolo export-onnx export-trt benchmark test lint

sync:
	uv sync

up-redis:
	docker compose up -d redis

up-vllm:
	docker compose --profile vllm up -d redis vllm

down-vllm:
	docker compose --profile vllm stop vllm

up-gguf:
	docker compose --profile gguf up -d redis llamacpp

up-mlflow:
	docker compose up -d mlflow

download-taco:
	uv run python scripts/download_taco.py --limit $(TACO_LIMIT)

enqueue:
	uv run python scripts/enqueue_jobs.py

reset-dlq:
	docker exec vlm2edge-redis redis-cli del vlm2edge:queue:dlq vlm2edge:queue:pending

worker:
	uv run python scripts/worker.py

eval:
	uv run python scripts/evaluate_labels.py

fiftyone:
	uv run --python 3.12 --with fiftyone python scripts/inspect_fiftyone.py

split-yolo:
	uv run python scripts/prepare_yolo_splits.py

train-yolo: down-vllm up-mlflow
	uv run --python 3.12 --with ultralytics --with "mlflow<3" python scripts/train_yolo.py

export-onnx:
	uv run --python 3.12 --with ultralytics --with onnx --with onnxslim python scripts/export_yolo.py --format onnx

export-trt:
	uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime-gpu --with tensorrt --with "nvidia-modelopt[onnx]>=0.44" python scripts/export_yolo.py --format all

benchmark:
	uv run --python 3.12 --with ultralytics --with onnxruntime-gpu --with tensorrt python scripts/benchmark_models.py

test:
	uv run pytest -v

lint:
	uv run ruff check src tests scripts
	uv run pyright
