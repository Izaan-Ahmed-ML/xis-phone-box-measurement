import cv2
import numpy as np
import glob
import os
COLS=7
ROWS=7
# --- constants you'll fill in based on your board ---
CHECKERBOARD = (COLS, ROWS)   # inner corners, e.g. (7, 5)
SQUARE_SIZE_MM = 48.25          # real-world size of one square

# # prepare known 3D object points, e.g. (0,0,0), (1,0,0), (2,0,0) ...
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp *= SQUARE_SIZE_MM

# objpoints = []   # 3D points, real world
# imgpoints = []   # 2D points, image plane

# script_dir = os.path.dirname(os.path.abspath(__file__))
# images_dir = os.path.join(script_dir, 'calibration_images')
# images = glob.glob(os.path.join(images_dir, '*.jpg'))
# print(f"Looking for images in: {images_dir}")
# print(f"Found {len(images)} image(s)")

# for fname in images:
#     img = cv2.imread(fname)
#     if img is None:
#         print(f"{fname}: failed to read image (None). Skipping.")
#         continue
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     except Exception as e:
#         print(f"{fname}: error converting to grayscale: {e}")
#         continue

#     try:
#         ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
#     except Exception as e:
#         print(f"{fname}: error finding chessboard corners: {e}")
#         continue

#     if ret:
#         objpoints.append(objp)
#         # refine corner locations to sub-pixel accuracy
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#         print(f"{fname}: corners found ✓")
#     else:
#         print(f"{fname}: corners NOT found ✗ — check angle/blur/lighting")

    
#     ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None,
#     flags=cv2.CALIB_FIX_K3

# )

# print("Reprojection error:", ret)
# print("Camera matrix (K):\n", K)
# print("Distortion coefficients:\n", distCoeffs)

# np.savez('D:\project-root\Calibration\camera_params.npz', K=K, distCoeffs=distCoeffs)


#if k is too high then checking symmetry ambiguty
# import cv2

# files_to_check = [
#     r'd:\project-root\Calibration\calibration_images\calib_03.jpg',
#     r'd:\project-root\Calibration\calibration_images\calib_10.jpg',
#     r'd:\project-root\Calibration\calibration_images\calib_15.jpg',
# ]

# for fname in files_to_check:
#     img = cv2.imread(fname)
#     if img is None:
#         print(f"FAILED TO LOAD: {fname}")
#         continue
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
#     if ret:
#         cv2.drawChessboardCorners(img, (7, 7), corners, ret)
#         out = fname.replace('.jpg', '_corners.jpg')
#         cv2.imwrite(out, img)
#         print(f"saved {out}")
#     else:
#         print(f"corners not found in {fname}")
# img = cv2.imread(r'D:\project-root\Calibration\calibration_images\calib_05.jpg')
# h, w = img.shape[:2]
# newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
# undistorted = cv2.undistort(img, K, distCoeffs, None, newK)
# cv2.imwrite(r'D:\project-root\Calibration\undistort_test.jpg', undistorted)
data = np.load('Calibration/camera_params.npz')
print(data['K'])
print(data['distCoeffs'])