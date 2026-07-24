
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


CALIB_PATH = "Calibration/camera_params.npz"
MODEL_PATH = "models/best_unet_resnet34.pth"
IMG_SIZE = 512  # must match training resolution


def load_calibration(calib_path):
    data = np.load(calib_path)
    return data['K'], data['distCoeffs']


def undistort_image(img, K, distCoeffs):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, distCoeffs, None, newK)
    return undistorted


def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
 )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor, (original_h, original_w)


def run_inference(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)
    return probs.squeeze().cpu().numpy()  # (H, W), values 0-1


def postprocess_mask(prob_map, original_size, threshold=0.5):
    mask = (prob_map > threshold).astype(np.uint8) * 255
    # Resize mask back up to the original image resolution
    mask_full_res = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_full_res


def create_overlay(original_img, mask, alpha=0.4, color=(0, 255, 0)):
    overlay = original_img.copy()
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask > 0] = color
    blended = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

    # Draw contour outline for clarity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)

    return blended


def main():
    parser = argparse.ArgumentParser(description="Phone box segmentation inference pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="inference/output.jpg", help="Path to save annotated output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 1. Load calibration
    K, distCoeffs = load_calibration(CALIB_PATH)
    print("Calibration loaded.")

    # 2. Load raw image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    # 3. Undistort
    undistorted = undistort_image(img, K, distCoeffs)
    print("Image undistorted.")

    # 4. Load model
    model = load_model(MODEL_PATH, device)
    print("Model loaded.")

    # 5. Preprocess
    tensor, original_size = preprocess(undistorted, IMG_SIZE)

    # 6. Run inference
    prob_map = run_inference(model, tensor, device)

    # 7. Postprocess mask back to full resolution
    mask = postprocess_mask(prob_map, original_size)

    # 8. Create overlay visualization
    result = create_overlay(undistorted, mask)

    # 9. Save outputs
    cv2.imwrite(args.output, result)
    mask_path = args.output.replace(".jpg", "_mask.jpg")
    cv2.imwrite(mask_path, mask)

    print(f"Saved annotated result to: {args.output}")
    print(f"Saved raw mask to: {mask_path}")


if __name__ == "__main__":
    main()