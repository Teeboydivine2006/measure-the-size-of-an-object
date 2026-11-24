from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --- Step 1: Load the YOLO model once at module level ---
MODEL_PATH = Path(__file__).resolve().parent / "model" / "yolo11n.pt"
MODEL = YOLO(MODEL_PATH)


def detect_objects(
    image: np.ndarray,
    target_class: str | None = None,
    conf_threshold: float = 0.5,
) -> list[dict]:
    """
    Detects objects in a BGR image using YOLOv11.
    Returns a list of dicts sorted by confidence (highest first):
        [{"box": (x1, y1, x2, y2), "label": str, "confidence": float}, ...]
    Returns an empty list [] if nothing matches.
    """
    if image is None or image.size == 0:
        return []

    # --- Step 3: Run inference ---
    results = MODEL.predict(source=image, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return []

    detections: list[dict] = []

    # --- Step 4: Loop through results and extract plain Python values ---
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = MODEL.names[class_id]

        # --- Step 5: Optional class filter ---
        if target_class is not None and label.lower() != target_class.lower():
            continue

        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # --- Step 2: Package into a clean dictionary ---
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": confidence,
            }
        )

    # --- Step 6: Sort highest-confidence first ---
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


# Quick standalone test (won't run when imported by Script 5)
if __name__ == "__main__":
    sample_img = cv2.imread("phone.jpg")
    matches = detect_objects(sample_img, target_class= None, conf_threshold=0.5)

    if not matches:
        print("No matching objects detected.")
    else:
        for obj in matches:
            print(f"{obj['label']} ({obj['confidence']:.2f}) -> box: {obj['box']}")