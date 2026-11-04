
# XIS Technical Assessment — Phone Box Measurement Pipeline

An end-to-end computer vision pipeline that calibrates a camera to remove
lens distortion, trains a custom U-Net segmentation model on a self-collected
dataset of a phone box, and computes real-world width/height measurements
(in mm) from segmented masks using an ArUco reference marker for pixel-to-mm
conversion.

## Project Overview

This project implements a full industrial-style measurement workflow:
1. Camera calibration to correct lens distortion
2. Custom dataset collection and polygon-based labelling
3. Segmentation model training (U-Net with ResNet34 encoder, transfer learning)
4. Pixel-to-millimetre measurement using a calibrated ArUco reference marker

**Key results:** Segmentation test-set IoU of 0.9867; measurement accuracy
with mean percentage error of ~11-12% (see MEASUREMENT_REPORT.md for full
error analysis).

## Quick Start

See [SETUP.md](docs/SETUP.md) for full installation instructions.

pip install -r requirements.txt
python inference/inference.py --image path/to/photo.jpg --output result.jpg
python measurement/measure_box.py path/to/photo_with_marker.jpg


## Repository Structure

project-root/
calibration/ # Camera calibration script, images, saved parameters
dataset/ # Dataset collection and undistortion scripts
models/ # Training notebook and model config (weights on Drive)
inference/ # Inference script (undistort -> segment -> overlay)
measurement/ # ArUco reference detection + pixel-to-mm pipeline
docs/ # All documentation and reports

## Documentation

- [CALIBRATION_REPORT.md](docs/Calibration_report.md) — camera calibration method, parameters, debugging process
- [DATASET_CARD.md](docs/Dataset_card.md) — dataset collection, labelling, splits
- [TRAINING_REPORT.md](docs/Training_report.md) — model architecture, training, metrics
- [MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) — measurement methodology, accuracy validation
- [INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) — inference script usage
- [SETUP.md](docs/SETUP.md) — full setup and installation

## Large Files (Google Drive)

Per submission guidelines, large files are hosted externally
(anyone with the link can view):

- **Calibration Images**: https://drive.google.com/drive/folders/1oHZE6fndolY465ghD-MWcZwH-EmkIaRv?usp=sharing
- **Dataset — Raw & Undistorted Images**: = https://drive.google.com/drive/folders/1uuo1a6oxlzBSg4lrItJ75zp0E5Bo0l-9?usp=sharing
- **Dataset — Labelled Export**: https://drive.google.com/drive/folders/1ggQ5v5RhrfJCYNOPfEuNagUY9e87KN7y?usp=sharing
- **Trained Model Weights** (best_unet_resnet34.pth): https://drive.google.com/file/d/15joSOae3agqeE6RzCTd5HSGqulEGN0yD/view?usp=sharing
- **Measurement Validation Photos** (10 instances):  https://drive.google.com/drive/folders/1hUGySAJu375e7jipCbjVKCuzZq5FhZcU?usp=sharing

## Known Limitations

- Measurement pipeline shows a consistent ~11-12% overestimation bias,
  root-caused to the segmentation model's learned boundary (see
  MEASUREMENT_REPORT.md for full diagnostic process).
- Training dataset limited to 88 images of a single object.
- Training performed on Google Colab GPU due to local hardware constraints
  (inference validated to run locally on CPU with no GPU dependency).

Full training pipeline (data loading, model setup, training loop, evaluation): 
`models/training/train_unet.ipynb` — designed to run on Google Colab with a GPU runtime.

