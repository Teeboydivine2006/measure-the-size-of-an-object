from pathlib import Path
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Step 2: Load config + checkpoint once at module level ---
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "model" / "sam2_b.pt"

SAM2_MODEL = build_sam2(MODEL_CFG, str(CHECKPOINT_PATH), device=DEVICE)
PREDICTOR = SAM2ImagePredictor(SAM2_MODEL)


def segment_object(
    image: np.ndarray,
    box: tuple[float, float, float, float] | list[float],
) -> dict | None:
    """
    Segments a single object inside an [x1, y1, x2, y2] bounding box using SAM 2.
    Returns:
        {"mask": np.ndarray (H, W) of bool, "score": float}
    Returns None if the image or box is invalid.
    """
    if image is None or image.size == 0 or box is None:
        return None

    #--- Step 1: Convert BGR to RGB for SAM 2 ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Step 3: Run the heavy image encoder once ---
    with torch.inference_mode(): #this tells encoder to not compute gradients, which saves memory and speeds up inference
        PREDICTOR.set_image(image_rgb)

        # --- Step 4 & 5: Pass the YOLO box prompt and request the single best mask ---
        box_array = np.array(box, dtype=np.float32)
        masks, scores, _ = PREDICTOR.predict(
            box=box_array,
            multimask_output=False,
        )

    # --- Step 6 & 7: Unwrap to a 2D (H, W) boolean mask + scalar confidence score ---
    mask = masks[0].astype(bool)
    score = float(scores[0])

    return {
        "mask": mask,
        "score": score,
    }


# Quick standalone test (won't run when imported by Script 5)
if __name__ == "__main__":
    sample_img = cv2.imread("test_image.jpg")
    sample_box = (100, 80, 320, 410)  # Example [x1, y1, x2, y2] box

    result = segment_object(sample_img, sample_box)
    if result is not None:
        print(f"Mask shape: {result['mask'].shape}, dtype: {result['mask'].dtype}")
        print(f"SAM 2 confidence score: {result['score']:.3f}")