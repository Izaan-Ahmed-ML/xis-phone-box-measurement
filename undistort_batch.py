import cv2
import numpy as np
import os
import glob

# --- Paths ---
# --- Paths ---
# Added 'r' before the strings to handle Windows backslashes properly
CALIB_FILE = r'D:\project-root\Calibration\camera_params.npz'
RAW_DIR = r'D:\project-root\dataset\dataset_box_cover'
OUTPUT_DIR = r'D:\project-root\dataset\dataset_box_cover\undistroted'

# --- Load calibration ---
if not os.path.exists(CALIB_FILE):
    raise FileNotFoundError(f"Calibration file not found at {CALIB_FILE} — run calibration first.")

data = np.load(CALIB_FILE)
K = data['K']
distCoeffs = data['distCoeffs']

print("Loaded calibration:")
print("K:\n", K)
print("distCoeffs:\n", distCoeffs)

# --- Create output folder if it doesn't exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Process every image ---
image_paths = glob.glob(os.path.join(RAW_DIR, '*.jpg')) + \
              glob.glob(os.path.join(RAW_DIR, '*.jpeg')) + \
              glob.glob(os.path.join(RAW_DIR, '*.png'))

print(f"\nFound {len(image_paths)} raw images to process.\n")

success_count = 0
fail_count = 0

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        print(f"FAILED TO LOAD: {path}")
        fail_count += 1
        continue

    h, w = img.shape[:2]

    # Compute optimal new camera matrix for THIS image's resolution
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))

    undistorted = cv2.undistort(img, K, distCoeffs, None, newK)

    filename = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, undistorted)

    print(f"✓ {filename} -> {out_path}")
    success_count += 1

print(f"\nDone. {success_count} undistorted, {fail_count} failed.")