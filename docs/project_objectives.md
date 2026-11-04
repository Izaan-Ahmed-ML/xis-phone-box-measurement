# Project Objectives

## Measurable Objectives

| ID | Field | Entry |
| :--- | :--- | :--- |
| **MO-1** | **What is measured** | Measurement accuracy of the phone box width and height (in mm) against physical caliper measurements across 10 validation instances. |
| | **How** | A measurement script compares system outputs against hand-measured caliper ground truth data. |
| | **Target** | Mean Percentage Error (MPE) documented and analyzed (observed baseline ~11-12% due to boundary mask expansion). |
| | **Conditions** | Locked phone camera focal length, indoor lighting, coplanar placement of phone box and ArUco reference marker. |
| | **Not included** | Extreme low-light conditions, severe camera tilt ($>30^\circ$), or partially occluded reference markers. |
| **MO-2** | **What is measured** | Segmentation model mask Intersection over Union (IoU) on the held-out test split. |
| | **How** | Evaluated via automated testing scripts on the 10% test split committed prior to model training. |
| | **Target** | Test-set mask IoU $\ge 0.95$ (achieved **0.9867**). |
| | **Conditions** | Undistorted inference images processed through the trained U-Net (ResNet34 backbone). |