import os
from pathlib import Path

from vlm2edge.eval.metrics import compute_precision_recall_f1, match_boxes_greedy
from vlm2edge.types import YoloBox


def load_yolo_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts: list[str] = line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append(
            YoloBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def main() -> None:
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "./data/taco"))
    gt_dir: Path = dataset_dir / "labels_gt"
    vlm_dir: Path = dataset_dir / "labels_vlm"

    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    evaluated_files: int = 0

    for vlm_file in sorted(vlm_dir.glob("*.txt")):
        gt_file: Path = gt_dir / vlm_file.name
        if not gt_file.exists():
            continue
        preds: list[YoloBox] = load_yolo_file(vlm_file)
        gts: list[YoloBox] = load_yolo_file(gt_file)
        tp, fp, fn = match_boxes_greedy(preds, gts, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        evaluated_files += 1

    metrics: dict[str, float] = compute_precision_recall_f1(total_tp, total_fp, total_fn)
    print(f"Evaluated images: {evaluated_files}")
    print(f"TP={total_tp} | FP={total_fp} | FN={total_fn}")
    print(
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
