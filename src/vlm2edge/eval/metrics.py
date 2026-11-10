from vlm2edge.types import YoloBox


def compute_iou(box_a: YoloBox, box_b: YoloBox) -> float:
    """
    Calcular o Intersection over Union (IoU) entre duas YoloBox normalizadas.

    Dica:
    1. Use `box_a.to_xyxy()` e `box_b.to_xyxy()` para obter (x1, y1, x2, y2).
    2. Calcule a área da interseção: max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1).
    3. Calcule a união: area_a + area_b - inter_area.
    4. Retorne inter_area / union_area (cuidado com divisão por zero).
    """

    ax1, ay1, ax2, ay2 = box_a.to_xyxy()
    bx1, by1, bx2, by2 = box_b.to_xyxy()

    if ax1 >= ax2 or ay1 >= ay2: return 0.0
    if bx1 >= bx2 or by1 >= by2: return 0.0
    
    inter_x1: float = max(ax1, bx1)
    inter_x2: float = min(ax2, bx2)

    inter_y1: float = max(ay1, by1)
    inter_y2: float = min(ay2, by2)

    inter_area: float = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    
    area_a: float = (ax2 - ax1) * (ay2 - ay1)
    area_b: float = (bx2 - bx1) * (by2 - by1)

    union_area: float = area_a + area_b - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_boxes_greedy(
    pred_boxes: list[YoloBox],
    gt_boxes: list[YoloBox],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Casar cada caixa predita pelo VLM com no máximo uma caixa do Ground Truth
    da mesma classe (`class_id`) cujo IoU >= `iou_threshold`.

    Retorna:
        (tp, fp, fn):
        - tp: número de caixas preditas que casaram com uma caixa GT livre
        - fp: caixas preditas sem par no GT
        - fn: caixas GT que não foram detectadas
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    matched_gt: set[int] = set()

    for pred_box in pred_boxes:
        best_gt_index: int = -1
        highest_iou: float = 0.0
    
        for i, gt_box in enumerate(gt_boxes):
            if i not in matched_gt and pred_box.class_id == gt_box.class_id:
                calc_iou: float = compute_iou(pred_box, gt_box)

                if calc_iou >= iou_threshold and calc_iou > highest_iou:
                    highest_iou = calc_iou
                    best_gt_index = i

        if best_gt_index != -1:
            matched_gt.add(best_gt_index)
            tp += 1

    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - tp

    return tp, fp, fn


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    Calcular precisão, recall e F1-score a partir de TP, FP e FN.
    Retornar dicionário com chaves: {"precision": ..., "recall": ..., "f1": ...}.
    """

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1_score: float = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1_score}