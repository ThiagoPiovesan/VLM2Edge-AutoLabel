from pydantic import BaseModel, Field


class YoloBox(BaseModel):
    """Normalized YOLO bounding box [x_center, y_center, width, height] in [0, 1]."""

    class_id: int = Field(ge=0)
    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert normalized YOLO center format to (x1, y1, x2, y2) in [0, 1]."""
        half_w: float = self.width / 2.0
        half_h: float = self.height / 2.0
        return (
            max(0.0, self.x_center - half_w),
            max(0.0, self.y_center - half_h),
            min(1.0, self.x_center + half_w),
            min(1.0, self.y_center + half_h),
        )

    def to_yolo_line(self) -> str:
        """Format box as a standard YOLO label file line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class LabelJob(BaseModel):
    """Job payload for auto-labeling a single image."""

    job_id: str
    image_path: str
    output_label_path: str
    attempt: int = 0
    last_error: str | None = None
