# Annotation Guide

## Bounding Box Annotation

Tool: CVAT (Computer Vision Annotation Tool)

### Classes
- **0: unripe** — USDA color stages 1 (Green), 2 (Breaker), 3 (Turning)
- **1: ripe** — USDA color stages 4 (Pink), 5 (Light Red), 6 (Red)

### Format
YOLO format: `class x_center y_center width height`
All coordinates normalized to [0, 1].

## Center-Point Annotation (Ripe Only)

### Definition
The center-point represents the **estimated full-fruit geometric center** —
the intersection of the fruit's horizontal and vertical symmetry axes.

### Protocol
1. Set CVAT zoom to minimum 200%
2. For unoccluded fruit: place keypoint at visual center of circular cross-section
3. For partially occluded fruit: mentally complete the circular boundary based
   on visible contour curvature; place keypoint at center of inferred circle
4. Format: append `cx cy` (normalized) to the YOLO annotation line

### Annotators
- A1 (primary): First author — used for all reported RMSE values
- A2 (validation): Trained research assistant — used for inter-annotator agreement

### Inter-Annotator Agreement
Euclidean RMSE: 2.18 px (1.78 mm at 500 mm)
<5px agreement: 94.4%
