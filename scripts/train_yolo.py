import argparse
import csv
import os
from pathlib import Path
from typing import Any


def load_existing_metrics(run_dir: Path) -> dict[str, float] | None:
    """Read best epoch metrics from an already completed Ultralytics results.csv."""
    csv_path: Path = run_dir / "results.csv"
    best_pt: Path = run_dir / "weights" / "best.pt"
    if not csv_path.exists() or not best_pt.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        rows: list[dict[str, str]] = [
            {k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)
        ]
    if not rows:
        return None
    best_row: dict[str, str] = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0.0)))
    return {
        "precision": float(best_row.get("metrics/precision(B)", 0.0)),
        "recall": float(best_row.get("metrics/recall(B)", 0.0)),
        "mAP50": float(best_row.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(best_row.get("metrics/mAP50-95(B)", 0.0)),
    }


def train_variant(
    variant: str,
    data_yaml: Path,
    model_weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project_dir: Path,
    force: bool = False,
) -> dict[str, float]:
    import mlflow  # type: ignore[import-untyped]
    from ultralytics import YOLO  # type: ignore[import-untyped]

    run_name: str = f"yolo-{variant}"
    run_dir: Path = project_dir.resolve() / run_name
    if not force:
        existing: dict[str, float] | None = load_existing_metrics(run_dir)
        if existing is not None:
            print(f"\n=== Skipping [{run_name}] (already completed at {run_dir}) ===")
            return existing

    print(f"\n=== Training [{run_name}] with data={data_yaml} ===")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    os.environ["MLFLOW_RUN"] = run_name

    model: Any = YOLO(model_weights)
    results: Any = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir.resolve()),
        name=run_name,
        exist_ok=True,
        seed=42,
        plots=True,
    )
    if mlflow.active_run() is not None:
        mlflow.end_run()

    results_dict: dict[str, Any] = getattr(results, "results_dict", {})
    return {
        "precision": float(results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO on GT vs VLM labels and log metrics to MLflow."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(os.getenv("DATASET_DIR", "./data/taco")),
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("./runs/compare"))
    parser.add_argument("--mode", choices=["both", "gt", "vlm"], default="both")
    parser.add_argument("--force", action="store_true", help="Retrain even if best.pt exists.")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "vlm2edge-taco")

    from ultralytics import settings  # type: ignore[import-untyped]

    settings.update({"mlflow": True})

    splits_dir: Path = args.dataset_dir / "yolo_splits"
    variants: list[str] = ["gt", "vlm"] if args.mode == "both" else [args.mode]
    summary: dict[str, dict[str, float]] = {}

    for variant in variants:
        yaml_path: Path = splits_dir / f"taco_{variant}" / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Missing {yaml_path}. Run `make split-yolo` before training."
            )
        summary[variant] = train_variant(
            variant=variant,
            data_yaml=yaml_path,
            model_weights=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project_dir=args.project,
            force=args.force,
        )

    print("\n================ NÍVEL 2: YOLO (GT) vs YOLO (VLM) ================")
    for variant, m in summary.items():
        print(
            f"[{variant.upper():3s}] Precision={m['precision']:.4f} | "
            f"Recall={m['recall']:.4f} | "
            f"mAP50={m['mAP50']:.4f} | "
            f"mAP50-95={m['mAP50_95']:.4f}"
        )
    if "gt" in summary and "vlm" in summary and summary["gt"]["mAP50"] > 0:
        retention: float = (summary["vlm"]["mAP50"] / summary["gt"]["mAP50"]) * 100.0
        print(f"-> VLM Auto-Label retém {retention:.1f}% do mAP50 do treino com GT humano!")


if __name__ == "__main__":
    main()
