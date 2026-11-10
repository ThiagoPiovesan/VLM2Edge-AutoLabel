import argparse
import time
from pathlib import Path
from typing import Any

from PIL import Image

from vlm2edge.eval.benchmark import compute_latency_stats


def measure_model_latencies(
    model_path: Path,
    loaded_images: list[Image.Image],
    device: str,
    imgsz: int = 640,
    warmup_runs: int = 10,
) -> list[float]:
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model: Any = YOLO(str(model_path), task="detect")
    warmup_img: Image.Image = loaded_images[0]
    for _ in range(warmup_runs):
        model.predict(source=warmup_img, device=device, imgsz=imgsz, verbose=False)

    latencies_ms: list[float] = []
    for img in loaded_images:
        t0: float = time.perf_counter()
        model.predict(source=img, device=device, imgsz=imgsz, verbose=False)
        elapsed_ms: float = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
    return latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO model latency (p50, p95, FPS) across formats/devices."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[
            Path("./runs/compare/yolo-vlm/weights/best.pt"),
            Path("./runs/compare/yolo-vlm/weights/best.onnx"),
            Path("./runs/compare/yolo-vlm/weights/best_fp16.engine"),
            Path("./runs/compare/yolo-vlm/weights/best_int8.engine"),
        ],
        help="List of model files (.pt, .onnx, .engine) to benchmark.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("./data/taco/yolo_splits/taco_vlm/images/val"),
    )
    parser.add_argument("--device", type=str, default="0", help="'0' for GPU or 'cpu'.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--eval-map", action="store_true", help="Also evaluate mAP50 on val split.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data/taco/yolo_splits/taco_vlm/dataset.yaml"),
    )
    args = parser.parse_args()

    images: list[Path] = sorted(args.images_dir.glob("*.jpg"))[: args.limit]
    if not images:
        raise FileNotFoundError(f"No validation images found in {args.images_dir}")

    # Pre-load and resize images in RAM so we benchmark inference, not WSL2 /mnt/c disk I/O
    loaded_images: list[Image.Image] = [
        Image.open(p).convert("RGB").resize((640, 640)) for p in images
    ]

    print(f"\nBenchmarking on {len(loaded_images)} in-memory images (device={args.device}):")
    print(f"{'Model':<40} | {'Mean (ms)':>10} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'FPS':>8}")
    print("-" * 90)
    for model_path in args.models:
        if not model_path.exists():
            print(f"{model_path.name:<40} | [SKIPPED - file not found]")
            continue
        lats: list[float] = measure_model_latencies(
            model_path=model_path,
            loaded_images=loaded_images,
            device=args.device,
        )
        stats: dict[str, float] = compute_latency_stats(lats)
        map_suffix: str = ""
        if args.eval_map and args.data.exists():
            from ultralytics import YOLO  # type: ignore[import-untyped]

            val_res: Any = YOLO(str(model_path), task="detect").val(
                data=str(args.data), device=args.device, batch=1, verbose=False
            )
            rdict: dict[str, Any] = getattr(val_res, "results_dict", {})
            map50: float = float(rdict.get("metrics/mAP50(B)", 0.0))
            map95: float = float(rdict.get("metrics/mAP50-95(B)", 0.0))
            map_suffix = f" | mAP50={map50:.4f} | mAP50-95={map95:.4f}"
        print(
            f"{model_path.name:<40} | "
            f"{stats['mean_ms']:>10.2f} | "
            f"{stats['p50_ms']:>10.2f} | "
            f"{stats['p95_ms']:>10.2f} | "
            f"{stats['fps']:>8.1f}{map_suffix}"
        )


if __name__ == "__main__":
    main()
