import pytest

from vlm2edge.eval.metrics import (
    compute_iou,
    compute_precision_recall_f1,
    match_boxes_greedy,
)
from vlm2edge.types import YoloBox


def test_compute_iou_identical() -> None:
    box: YoloBox = YoloBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4)
    assert compute_iou(box, box) == pytest.approx(1.0, rel=1e-5)


def test_compute_iou_disjoint() -> None:
    box_a: YoloBox = YoloBox(class_id=0, x_center=0.2, y_center=0.2, width=0.1, height=0.1)
    box_b: YoloBox = YoloBox(class_id=0, x_center=0.8, y_center=0.8, width=0.1, height=0.1)
    assert compute_iou(box_a, box_b) == pytest.approx(0.0, abs=1e-6)


def test_compute_iou_half_overlap() -> None:
    box_a: YoloBox = YoloBox(class_id=0, x_center=0.4, y_center=0.5, width=0.4, height=0.4)
    box_b: YoloBox = YoloBox(class_id=0, x_center=0.6, y_center=0.5, width=0.4, height=0.4)
    # x ranges: [0.2, 0.6] and [0.4, 0.8] -> overlap 0.2; union width 0.6 -> IoU = 0.2 / 0.6 = 1/3
    assert compute_iou(box_a, box_b) == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_match_boxes_and_metrics() -> None:
    gt: list[YoloBox] = [
        YoloBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2),
        YoloBox(class_id=0, x_center=0.2, y_center=0.2, width=0.1, height=0.1),
    ]
    preds: list[YoloBox] = [
        YoloBox(class_id=0, x_center=0.51, y_center=0.5, width=0.2, height=0.2),  # TP
        YoloBox(class_id=0, x_center=0.85, y_center=0.85, width=0.1, height=0.1),  # FP
    ]
    tp, fp, fn = match_boxes_greedy(preds, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 1)

    metrics: dict[str, float] = compute_precision_recall_f1(tp, fp, fn)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_match_boxes_duplicate_preds_and_best_iou() -> None:
    """Two GTs and three preds (one duplicate): both GTs must matcheven if 2nd has lower IoU."""
    gt: list[YoloBox] = [
        YoloBox(class_id=0, x_center=0.2, y_center=0.2, width=0.2, height=0.2),
        YoloBox(class_id=0, x_center=0.8, y_center=0.8, width=0.2, height=0.2),
    ]
    preds: list[YoloBox] = [
        YoloBox(class_id=0, x_center=0.2, y_center=0.2, width=0.2, height=0.2),   # IoU 1.0 with GT[0] -> TP
        YoloBox(class_id=0, x_center=0.21, y_center=0.2, width=0.2, height=0.2),  # duplicate on GT[0] -> FP
        YoloBox(class_id=0, x_center=0.82, y_center=0.8, width=0.2, height=0.2),  # IoU ~0.82 with GT[1] -> TP
    ]
    tp, fp, fn = match_boxes_greedy(preds, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (2, 1, 0)


def test_compute_precision_recall_f1_zero_division() -> None:
    """Zero TP or empty inputs must return 0.0 without raising ZeroDivisionError."""
    metrics_zero_tp: dict[str, float] = compute_precision_recall_f1(tp=0, fp=3, fn=2)
    assert metrics_zero_tp == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    metrics_all_zero: dict[str, float] = compute_precision_recall_f1(tp=0, fp=0, fn=0)
    assert metrics_all_zero == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

