"""
measurement/sanity_check_marker.py
Checks whether the ArUco marker measures close to its known true size,
using the exact same undistortion + detection + minAreaRect pipeline
as the main measurement script. This isolates whether errors come from
the reference/geometry pipeline or from the box's segmentation mask.
"""

import cv2
import numpy as np

MARKER_SIZE_MM = 165  # your confirmed true marker size
CALIB_PATH = "Calibration/camera_params.npz"
IMAGE_PATH = "measurement/test_shots/marker_and_box_01.jpeg"  # adjust if needed


def check_marker_self_measurement(image_path, calib_path, marker_size_mm=MARKER_SIZE_MM):
    data = np.load(calib_path)
    K, distCoeffs = data['K'], data['distCoeffs']

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, distCoeffs, None, newK)

    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        print("ERROR: No marker detected.")
        return

    marker_corners = corners[0][0]
    side_lengths = [np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4]) for i in range(4)]
    avg_side_px = np.mean(side_lengths)
    pixels_per_mm = avg_side_px / marker_size_mm

    # Now measure the marker's OWN size using minAreaRect, same method used for the box
    rect = cv2.minAreaRect(marker_corners.astype(np.float32))
    (w_px, h_px) = rect[1]

    print(f"pixels_per_mm: {pixels_per_mm:.4f}")
    print(f"Marker minAreaRect size: {w_px/pixels_per_mm:.1f}mm x {h_px/pixels_per_mm:.1f}mm")
    print(f"(True size is {marker_size_mm}mm x {marker_size_mm}mm — compare against this)")


if __name__ == "__main__":
    check_marker_self_measurement(IMAGE_PATH, CALIB_PATH)