# Definition of Done

- [x] **Camera Calibration:** 20+ calibration images processed via OpenCV; camera parameters saved; reprojection error reported in `docs/CALIBRATION_REPORT.md`.
- [x] **Dataset & Splits:** 88 raw/undistorted images collected, labelled, and split with a fixed train/val/test file list committed to the repository.
- [x] **Model Training:** Custom U-Net (ResNet34 encoder) trained; weights linked externally with SHA-256 hash in README; metrics (IoU: 0.9867) reported in `docs/TRAINING_REPORT.md`.
- [x] **Inference Command:** Single CLI command (`python inference/inference.py`) successfully executes undistortion $\rightarrow$ segmentation $\rightarrow$ ArUco detection $\rightarrow$ metric output with visual overlay and confidence score.
- [x] **Accuracy Validation:** 10 validation instances measured with a physical caliper and compared against system outputs in a complete error analysis table (`docs/MEASUREMENT_REPORT.md`).
- [x] **Undistortion Proof:** Side-by-side error comparison demonstrating the measurable impact of applying camera calibration vs. raw images.
- [x] **Error Handling & Thresholds:** Graceful handling and refusal logic implemented for cases where the object or reference marker is missing or below confidence thresholds.
- [x] **Documentation & Repository Structure:** Plan, Objectives, Definition of Done, and README properly formatted under `docs/` and root directory on branch `dev-test01`.