import cv2
import math
import numpy as np

#this is the specified patterns that opencv will be searching for in the image
pattern = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

#creates a detector parameters seting object
settings = cv2.aruco.DetectorParameters()

#here we create the detector object, which will be used to detect the patterns in the image
detector =cv2.aruco.ArucoDetector(pattern, settings)



def get_pixels_per_cm(image: np.ndarray, marker_size_cm: float = 5.0) -> float | None:
    """
    Detects an ArUco marker in a BGR image and calculates the pixels-per-cm ratio.
    Returns None if the image is invalid or no marker is found.
    """

    if image is None or image.size == 0:
        print("Error: Image not found or unable to load.")
        return None
    
    #convert the image to grayscale, as the detector works better on grayscale images
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detect the patterns in the image, and get the corners of the detected patterns
    corners, ids, rejected = detector.detectMarkers(gray_image)

    if ids is None or len(corners) == 0:
        print("No patterns detected in the image.")
        return None
    else:
        #extract the first detected pattern's corners
        first_pattern_corners = corners[0][0]
        tl, tr, br, bl = first_pattern_corners

        #calculate the edges of the detected pattern in pixels
        top_edge = math.dist(tl, tr)
        right_edge = math.dist(tr, br)
        bottom_edge = math.dist(br, bl)
        left_edge = math.dist(bl, tl)

        #calculate the average edge length in pixels
        avg_edge_length = (top_edge + right_edge + bottom_edge + left_edge) / 4

        print(f"Detected Marker ID: {ids[0]}")
        print(f"Average marker side length: {avg_edge_length:.2f} pixels")

        pixels_per_cm = avg_edge_length / marker_size_cm

        return pixels_per_cm

if __name__ == "__main__":
    #load the image that we want to detect the patterns in
    image = cv2.imread("phone_with_marker.jpg")
    pixels_per_cm = get_pixels_per_cm(image)
    if pixels_per_cm is not None:
        print(f"Pixels per cm: {pixels_per_cm:.2f}")