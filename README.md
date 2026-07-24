\# XIS Technical Assessment — Mobile Phone Box Measurement Pipeline



An end-to-end computer vision pipeline that calibrates a camera to remove

lens distortion, trains a custom segmentation model on a self-collected

dataset of Mobile Phone Box Measurement, and computes real-world width/height

measurements (in mm) from segmented masks using a calibrated

pixel-to-mm reference. Work in progress — see commit history for

incremental progress.

Dataset link:https://drive.google.com/drive/folders/1ggQ5v5RhrfJCYNOPfEuNagUY9e87KN7y?usp=sharing

Trained Model Weights: (U-Net + ResNet34, best checkpoint): https://drive.google.com/file/d/15joSOae3agqeE6RzCTd5HSGqulEGN0yD/view?usp=sharing

Full training pipeline (data loading, model setup, training loop, evaluation): 
`models/training/train_unet.ipynb` — designed to run on Google Colab with a GPU runtime.

