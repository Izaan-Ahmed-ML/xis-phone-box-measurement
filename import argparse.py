import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


CALIB_PATH = "D:/project-root/Calibration/camera_params.npz"
MODEL_PATH = "D:/project-root/models/best_unet_resnet34.pth"
IMG_SIZE = 512          # must match training resolution
MARKER_SIZE_MM = 165    # printed ArUco marker side length (DICT_4X4_50)
MIN_BOX_AREA_PX = 500   # ignore tiny noise blobs


def load_calibration(calib_path):
    data = np.load(calib_path)
    return data["K"], data["distCoeffs"]


def build_undistort_maps(K, distCoeffs, w, h):
    """Compute the undistortion maps ONCE; remap() per frame is much faster than undistort()."""
    newK, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, None, newK, (w, h), cv2.CV_16SC2)
    return map1, map2, newK


def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(img, img_size):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0  # same as training (no mean/std)
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def run_inference(model, tensor, device):
    logits = model(tensor.to(device))
    probs = torch.sigmoid(logits)
    return probs.squeeze().cpu().numpy()


def postprocess_mask(prob_map, original_size, threshold=0.5):
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)


def create_overlay(original_img, mask, alpha=0.4, color=(0, 255, 0)):
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask > 0] = color
    blended = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)
    return blended


def make_marker_detector():
    """Create the ArUco detector ONCE (requires OpenCV >= 4.7)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def detect_reference_marker(detector, img, marker_size_mm):
    """Returns (pixels_per_mm, marker_corners (4,2)) or (None, None) if no marker is visible."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, None
    marker_corners = corners[0][0]
    sides = [np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides)) / marker_size_mm, marker_corners


def measure_box(mask):
    """Rotated bounding rectangle of the largest blob. Returns (long_px, short_px, corners)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BOX_AREA_PX:
        return None
    rect = cv2.minAreaRect(c)  # ((cx, cy), (w, h), angle)
    w, h = rect[1]
    corners = cv2.boxPoints(rect).astype(int)
    return max(w, h), min(w, h), corners


def parse_source(s):
    # "0" -> webcam index 0; anything else is treated as a path / RTSP / HTTP URL
    return int(s) if s.isdigit() else s


def main():
    parser = argparse.ArgumentParser(description="Live phone box segmentation + measurement")
    parser.add_argument("--source", default="0", help="Webcam index (0,1,...), video file, or stream URL")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-undistort", action="store_true", help="Skip undistortion")
    parser.add_argument("--record", default=None, help="Optional path to save output video (e.g. out.mp4)")
    parser.add_argument("--display-scale", type=float, default=0.6,
                        help="Scale of the preview window only (e.g. 0.5 = half size). "
                             "Processing and --record stay at full resolution.")
    # Scale options. Priority: --mm-per-px > ArUco marker > --distance-mm
    parser.add_argument("--no-marker", action="store_true", help="Disable ArUco marker scale detection")
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM,
                        help="Side length of the printed ArUco marker in mm")
    parser.add_argument("--mm-per-px", type=float, default=None,
                        help="Manual scale in mm per pixel (overrides marker and distance)")
    parser.add_argument("--distance-mm", type=float, default=None,
                        help="Fallback: distance from camera to the TOP face of the box, in mm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = load_model(MODEL_PATH, device)
    print("Model loaded.")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # IMPORTANT: calibration was done at a specific resolution. The camera must
    # deliver the same resolution, otherwise K is wrong and the undistortion is off.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: {w}x{h}")

    map1 = map2 = None
    fx = None
    if not args.no_undistort or args.distance_mm:
        K, distCoeffs = load_calibration(CALIB_PATH)
        print("Calibration loaded.")
        if not args.no_undistort:
            map1, map2, newK = build_undistort_maps(K, distCoeffs, w, h)
            fx = newK[0, 0]  # frames are remapped with newK, so use its focal length
        else:
            fx = K[0, 0]

    # Fallback scale from camera distance (pinhole model: size_mm = size_px * Z / fx)
    distance_scale = args.distance_mm / fx if (args.distance_mm and fx) else None

    use_marker = not args.no_marker and args.mm_per_px is None
    detector = make_marker_detector() if use_marker else None
    ppm_hist = deque(maxlen=15)  # recent pixels-per-mm readings from the marker
    marker_ppm = None            # last known (smoothed) pixels-per-mm; held if marker is covered

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w, h))

    # Warm-up (first CUDA call is slow)
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))

    hist = deque(maxlen=10)  # recent box measurements for median smoothing
    fps, prev = 0.0, time.time()
    print("Press 'q' or ESC to quit, 's' to save a snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No more frames / camera disconnected.")
            break

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # --- Scale from ArUco marker (detected on the clean undistorted frame) ---
        marker_seen = False
        marker_corners = None
        if use_marker:
            ppm, marker_corners = detect_reference_marker(detector, frame, args.marker_size_mm)
            if ppm is not None:
                ppm_hist.append(ppm)
                marker_ppm = float(np.median(ppm_hist))
                marker_seen = True

        # --- Segmentation ---
        tensor = preprocess(frame, IMG_SIZE)
        prob_map = run_inference(model, tensor, device)
        mask = postprocess_mask(prob_map, frame.shape[:2], args.threshold)
        result = create_overlay(frame, mask)

        if marker_seen:
            cv2.aruco.drawDetectedMarkers(result, [marker_corners.reshape(1, 4, 2)])

        # --- Pick the scale: manual > marker > distance > none ---
        if args.mm_per_px:
            mm_per_px, src = args.mm_per_px, "manual"
        elif marker_ppm:
            mm_per_px = 1.0 / marker_ppm
            src = "marker" if marker_seen else "marker (held)"
        elif distance_scale:
            mm_per_px, src = distance_scale, "distance"
        else:
            mm_per_px, src = None, "none"

        # --- Width / height ---
        m = measure_box(mask)
        if m:
            long_px, short_px, corners = m
            hist.append((long_px, short_px))
            long_med = np.median([a for a, _ in hist])
            short_med = np.median([b for _, b in hist])
            cv2.polylines(result, [corners], True, (255, 0, 0), 2)
            if mm_per_px:
                txt = f"{long_med * mm_per_px:.1f} x {short_med * mm_per_px:.1f} mm"
            else:
                txt = f"{long_med:.0f} x {short_med:.0f} px"
            cv2.putText(result, txt, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            hist.clear()
        cv2.putText(result, f"scale: {src}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS (smoothed)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Preview window (resized copy only; processing/recording stay full-res)
        if args.display_scale != 1.0:
            shown = cv2.resize(result, None, fx=args.display_scale, fy=args.display_scale,
                               interpolation=cv2.INTER_AREA)
        else:
            shown = result
        cv2.imshow("Live segmentation", shown)
        if writer is not None:
            writer.write(result)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(name, result)
            cv2.imwrite(name.replace(".jpg", "_mask.jpg"), mask)
            print(f"Saved {name}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()