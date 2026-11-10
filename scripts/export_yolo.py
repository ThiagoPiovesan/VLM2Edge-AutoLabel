import argparse
import shutil
from pathlib import Path
from typing import Any


def export_single(
    weights: Path,
    fmt: str,
    imgsz: int,
    half: bool,
    int8: bool,
    data: Path,
    output_name: str | None = None,
) -> Path:
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model: Any = YOLO(str(weights))
    exported_str: str = model.export(
        format=fmt,
        imgsz=imgsz,
        half=half,
        int8=int8,
        data=str(data) if int8 else None,
    )
    exported_path: Path = Path(exported_str)
    if output_name is not None:
        target_path: Path = exported_path.with_name(output_name)
        if exported_path.resolve() != target_path.resolve():
            shutil.move(str(exported_path), str(target_path))
            exported_path = target_path
    print(f"[OK] Exported -> {exported_path}")
    return exported_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained YOLO weights to ONNX and TensorRT (.engine FP16/INT8)."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("./runs/compare/yolo-vlm/weights/best.pt"),
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "engine", "all"],
        default="all",
        help="Target export format: 'onnx', 'engine', or 'all' (ONNX + TRT FP16 + TRT INT8).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Enable FP16 quantization.")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data/taco/yolo_splits/taco_vlm/dataset.yaml"),
        help="Dataset YAML required for INT8 calibration.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Trained weights not found at {args.weights}")

    stem: str = args.weights.stem
    if args.format == "all":
        export_single(args.weights, "onnx", args.imgsz, half=False, int8=False, data=args.data)
        export_single(
            args.weights,
            "engine",
            args.imgsz,
            half=True,
            int8=False,
            data=args.data,
            output_name=f"{stem}_fp16.engine",
        )
        export_single(
            args.weights,
            "engine",
            args.imgsz,
            half=False,
            int8=True,
            data=args.data,
            output_name=f"{stem}_int8.engine",
        )
    else:
        suffix: str = "_int8.engine" if args.int8 else ("_fp16.engine" if args.half else ".engine")
        out_name: str | None = f"{stem}{suffix}" if args.format == "engine" else None
        export_single(
            args.weights,
            args.format,
            args.imgsz,
            half=args.half,
            int8=args.int8,
            data=args.data,
            output_name=out_name,
        )


if __name__ == "__main__":
    main()
