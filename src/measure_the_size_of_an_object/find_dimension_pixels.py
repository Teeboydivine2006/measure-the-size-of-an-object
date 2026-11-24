
import cv2
import numpy as np

def get_pixels_dimensions(mask: np.ndarray) -> dict[str, float | np.ndarray] | None:
    """
    Given a binary mask of an object, returns the width and height in pixels.
    Returns None if the mask is invalid or empty.
    """
    if mask is None or mask.size == 0:
        return None

    # convert the mask to a binary image
    binary_mask = mask.astype(np.uint8) * 255

    # find contours in the binary mask
    contours, _ =cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #protect against empty contours
    if not contours:
        return None

    # get the largest contour (assuming it's the object)
    largest_contour = max(contours, key=cv2.contourArea)

    #dealing with the case where the contour area is zero
    if cv2.contourArea(largest_contour) == 0:
        return None

    #get the bounding rectangle of the largest contour
    rectangle = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (width, height), angle = rectangle

    #return the width and height as floats and also preventing a case where opencv swap the width and height if the object is rotated, so we take the max as height and min as width
    height_px = float(max(width, height))
    width_px = float(min(width, height))

    # Bonus for visual debugging: 4 corner coordinates of the rotated box
    box_points = np.int32(cv2.boxPoints(rectangle))

    return {
        "width_px": width_px,
        "height_px": height_px,
        "angle": float(angle),
        "box_points": box_points,
    }

# for testing the function independently
if __name__ == "__main__":
    # Create a synthetic 500x500 mask with a 100px x 200px rectangle tilted at 30°
    dummy_mask = np.zeros((500, 500), dtype=bool)
    tilted_rect = ((250, 250), (100, 200), 30.0)
    pts = np.int32(cv2.boxPoints(tilted_rect))
    cv2.fillPoly(dummy_mask.view(np.uint8), [pts], 1)

    dims = get_pixels_dimensions(dummy_mask)
    if dims is None:
        print("No valid object contour found in mask.")
    else:
        print(f"Width (shorter side):  {dims['width_px']:.2f} px")
        print(f"Height (longer side):  {dims['height_px']:.2f} px")
        print(f"Rotation angle:        {dims['angle']:.2f} deg")
        print(f"Rotated box corners:\n{dims['box_points']}")