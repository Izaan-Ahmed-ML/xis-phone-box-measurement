"""
measurement/measure_box.py
Full pipeline: undistort -> detect reference marker -> segment box -> measure in mm
"""

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

MARKER_SIZE_MM = 165
CALIB_PATH = "Calibration/camera_params.npz"
MODEL_PATH = "models/best_unet_resnet34.pth"
IMG_SIZE = 512


def load_calibration(calib_path):
    data = np.load(calib_path)
    return data['K'], data['distCoeffs']


def undistort_image(img, K, distCoeffs):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    return cv2.undistort(img, K, distCoeffs, None, newK)


def detect_reference_marker(img, marker_size_mm=MARKER_SIZE_MM):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    marker_corners = corners[0][0]
    side_lengths = [np.linalg.norm(marker_corners[i] - marker_corners[(i+1) % 4]) for i in range(4)]
    avg_side_px = np.mean(side_lengths)
    return avg_side_px / marker_size_mm


def load_model(model_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def segment_box(model, img, device, img_size=IMG_SIZE):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))
    prob_map = probs.squeeze().cpu().numpy()

    mask = (prob_map > 0.5).astype(np.uint8) * 255
    mask_full = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("measurement/debug_raw_mask.jpg", mask_full)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    kernel_open = np.ones((9, 9), np.uint8)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)
    cv2.imwrite("measurement/debug_cleaned_mask.jpg", mask_clean)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    mask_final = np.zeros_like(mask_full)
    cv2.drawContours(mask_final, [hull], -1, 255, thickness=cv2.FILLED)

    # --- NEW: erode the mask to correct for consistent over-segmentation ---
    kernel_erode = np.ones((7, 7), np.uint8)
    mask_eroded = cv2.erode(mask_final, kernel_erode, iterations=1)
    cv2.imwrite("measurement/debug_eroded_mask.jpg", mask_eroded)

    # Re-extract the contour from the eroded mask — this is what gets measured now
    eroded_contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not eroded_contours:
        return None, None
    final_contour = max(eroded_contours, key=cv2.contourArea)

    return mask_eroded, final_contour   # <-- return hull, not largest, so measure_box() uses it too


def measure_box(contour, pixels_per_mm):
    rect = cv2.minAreaRect(contour)  # ((cx,cy),(w,h),angle)
    (w_px, h_px) = rect[1]
    width_mm = min(w_px, h_px) / pixels_per_mm
    height_mm = max(w_px, h_px) / pixels_per_mm
    return width_mm, height_mm, rect


def main(image_path, output_path="measurement/measured_output.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K, distCoeffs = load_calibration(CALIB_PATH)
    img = cv2.imread(image_path)
    undistorted = undistort_image(img, K, distCoeffs)

    pixels_per_mm = detect_reference_marker(undistorted)
    if pixels_per_mm is None:
        print("ERROR: No reference marker detected.")
        return
    print(f"pixels_per_mm: {pixels_per_mm:.4f}")

    model = load_model(MODEL_PATH, device)
    mask, contour = segment_box(model, undistorted, device)
    if contour is None:
        print("ERROR: No box detected in segmentation.")
        return

    width_mm, height_mm, rect = measure_box(contour, pixels_per_mm)
    print(f"Measured width:  {width_mm:.1f} mm")
    print(f"Measured height: {height_mm:.1f} mm")

    box_pts = cv2.boxPoints(rect).astype(int)
    result = undistorted.copy()
    cv2.drawContours(result, [box_pts], 0, (0, 255, 0), 3)
    text = f"W: {width_mm:.1f}mm  H: {height_mm:.1f}mm"
    cv2.putText(result, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "measurement/test_shots/marker_and_box_01.jpeg")