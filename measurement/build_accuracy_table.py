"""
measurement/build_accuracy_table.py
Runs the full measurement pipeline on all validation photos,
compares against ground truth, and outputs an accuracy table + MAE/MPE.
"""

import csv
import cv2
import numpy as np
import torch
from measure_box import (
    load_calibration, undistort_image, detect_reference_marker,
    load_model, segment_box, measure_box, CALIB_PATH, MODEL_PATH
)

GROUND_TRUTH_CSV = "measurement/ground_truth.csv"
VALIDATION_DIR = "measurement/validation_shots"
OUTPUT_MD = "measurement/accuracy_table.md"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, distCoeffs = load_calibration(CALIB_PATH)
    model = load_model(MODEL_PATH, device)

    rows = []
    width_errors, height_errors = [], []
    width_pcts, height_pcts = [], []

    with open(GROUND_TRUTH_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row['instance']
            filename = row['filename']
            gt_w = float(row['gt_width_mm'])
            gt_h = float(row['gt_height_mm'])

            img_path = f"{VALIDATION_DIR}/{filename}"
            img = cv2.imread(img_path)
            if img is None:
                print(f"WARNING: could not load {img_path}, skipping")
                continue

            undistorted = undistort_image(img, K, distCoeffs)
            pixels_per_mm = detect_reference_marker(undistorted)
            if pixels_per_mm is None:
                print(f"WARNING: no marker detected in {filename}, skipping")
                continue

            mask, contour = segment_box(model, undistorted, device)
            if contour is None:
                print(f"WARNING: no box detected in {filename}, skipping")
                continue

            meas_w, meas_h, _ = measure_box(contour, pixels_per_mm)

            w_ae = abs(gt_w - meas_w)
            h_ae = abs(gt_h - meas_h)
            w_pe = (w_ae / gt_w) * 100
            h_pe = (h_ae / gt_h) * 100

            width_errors.append(w_ae)
            height_errors.append(h_ae)
            width_pcts.append(w_pe)
            height_pcts.append(h_pe)

            rows.append((instance, gt_w, meas_w, w_ae, w_pe, gt_h, meas_h, h_ae, h_pe))
            print(f"Instance {instance}: W {meas_w:.1f}mm (GT {gt_w}), H {meas_h:.1f}mm (GT {gt_h})")

    if not rows:
        print("No valid results — check filenames, marker visibility, and box detection.")
        return

    mae_w = np.mean(width_errors)
    mae_h = np.mean(height_errors)
    mpe_w = np.mean(width_pcts)
    mpe_h = np.mean(height_pcts)

    with open(OUTPUT_MD, 'w') as f:
        f.write("| Instance | GT Width (mm) | Measured W (mm) | AE (mm) | PE (%) | GT Height (mm) | Measured H (mm) | AE (mm) | PE (%) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.0f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f}% | {r[5]:.0f} | {r[6]:.1f} | {r[7]:.1f} | {r[8]:.1f}% |\n")
        f.write(f"\n**Mean Absolute Error:** Width {mae_w:.1f}mm, Height {mae_h:.1f}mm\n")
        f.write(f"**Mean Percentage Error:** Width {mpe_w:.1f}%, Height {mpe_h:.1f}%\n")

    print(f"\nTable written to {OUTPUT_MD}")
    print(f"Overall MAE: W={mae_w:.1f}mm, H={mae_h:.1f}mm")
    print(f"Overall MPE: W={mpe_w:.1f}%, H={mpe_h:.1f}%")


if __name__ == "__main__":
    main()