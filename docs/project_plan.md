# Project Plan: Phone Box Measurement Pipeline

## Project Overview
An end-to-end industrial computer vision pipeline that calibrates a camera to remove lens distortion, collects and polygon-labels a custom image dataset of a phone box, trains a U-Net segmentation model with a ResNet34 encoder, and computes real-world width and height measurements in millimeters from segmentation masks using an ArUco reference marker.

## Execution Timeline (8-Hour Allocation)

* **Hour 1 — Environment & Camera Calibration:** 
  * Initialize repository structure (`calibration/`, `dataset/`, `models/`, `inference/`, `measurement/`, `docs/`).
  * Capture 20+ calibration checkerboard/ChArUco images with locked focus, zoom, and resolution.
  * Run OpenCV camera calibration to generate and save camera matrices and distortion coefficients, reporting reprojection error.
* **Hours 2–3 — Dataset Collection & Annotation:** 
  * Capture ~88 images of the phone box under varied conditions using the calibrated camera.
  * Apply camera undistortion to all raw images.
  * Perform polygon annotation and commit the fixed train/val/test split file list.
* **Hours 4–5 — Model Training & Evaluation:** 
  * Set up a custom U-Net architecture with a ResNet34 pretrained encoder (avoiding forbidden YOLO/Roboflow models).
  * Train on Google Colab GPU runtime and evaluate performance metrics (IoU, precision, recall, mAP) on the held-out test split.
* **Hours 6–7 — Measurement Pipeline & ArUco Integration:** 
  * Implement ArUco marker detection script (`measurement/measure_box.py`) to derive the pixel-to-millimeter scaling ratio.
  * Build the unified inference command (`inference/inference.py`) handling undistortion, segmentation, reference detection, and mm conversion with visual overlays.
  * Validate accuracy against manual caliper measurements across 10 test instances.
* **Hour 8 — Documentation & Error Analysis:** 
  * Compile technical reports (`CALIBRATION_REPORT.md`, `DATASET_CARD.md`, `TRAINING_REPORT.md`, `MEASUREMENT_REPORT.md`).
  * Analyze measurement errors (e.g., boundary bias leading to overestimation) and define confidence refusal thresholds.
  * Finalize Git commit history and record the 5-minute Loom walkthrough.

## Decision Record
* **Architecture Selection:** Chosen U-Net with ResNet34 encoder for fine boundary segmentation over YOLO segmentation models to comply with project constraints.
* **Reference Marker:** Selected ArUco markers over standard planar reference objects due to robust corner identification and pose detection in OpenCV.