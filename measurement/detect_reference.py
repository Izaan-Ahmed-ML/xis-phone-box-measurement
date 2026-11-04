"""
measurement/detect_reference.py
Detects an ArUco reference marker in an image and computes pixels-per-mm.
"""

import cv2
import numpy as np

MARKER_SIZE_MM = 165  # confirmed via printed A4 sheet measurement


def load_calibration(calib_path):
    data = np.load(calib_path)
    return data['K'], data['distCoeffs']


def undistort_image(img, K, distCoeffs):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, distCoeffs, None, newK)
    return undistorted


def detect_reference_marker(img, marker_size_mm=MARKER_SIZE_MM):
    """
    Detects an ArUco marker in the image and returns pixels_per_mm ratio.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        print("WARNING: No ArUco marker detected in image.")
        return None, None

    marker_corners = corners[0][0]  # (4, 2) array of corner points

    side_lengths = []
    for i in range(4):
        pt1 = marker_corners[i]
        pt2 = marker_corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(pt1 - pt2))

    avg_side_px = np.mean(side_lengths)
    pixels_per_mm = avg_side_px / marker_size_mm

    print(f"Marker detected. Avg side length: {avg_side_px:.2f}px")
    print(f"pixels_per_mm: {pixels_per_mm:.4f}")

    return pixels_per_mm, marker_corners





if __name__ == "__main__":
    CALIB_PATH = "D:/project-root/Calibration/camera_params.npz"
    IMAGE_PATH = "D:/project-root/measurement/test_shots/marker_and_box_01.jpeg"  # adjust to your actual filename

    K, distCoeffs = load_calibration(CALIB_PATH)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

    undistorted = undistort_image(img, K, distCoeffs)
    pixels_per_mm, corners = detect_reference_marker(undistorted)

    if pixels_per_mm is not None:
        # Draw detected corners for visual confirmation
        cv2.aruco.drawDetectedMarkers(undistorted, [corners.reshape(1, 4, 2)])
        cv2.imwrite("measurement/detection_check.jpg", undistorted)
        print("Saved visual confirmation to measurement/detection_check.jpg")