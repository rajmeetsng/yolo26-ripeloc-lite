"""Center-Point Localization (CPL) RMSE evaluation."""
import argparse, json, torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_gt_centers(gt_dir, image_names):
    """Load ground-truth center points from annotation files."""
    centers = {}
    for name in image_names:
        txt = Path(gt_dir) / f"{Path(name).stem}.txt"
        if txt.exists():
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and int(parts[0]) == 1:  # ripe class
                        cx_gt = float(parts[5])
                        cy_gt = float(parts[6])
                        centers.setdefault(name, []).append((cx_gt, cy_gt))
    return centers

def compute_cpl_metrics(pred_centers, gt_centers, img_w=640, img_h=640,
                        focal_length=612.36, working_dist=500):
    """Compute RMSE, MAE, <5px accuracy, physical error."""
    errors_x, errors_y, errors_euc = [], [], []
    
    for (px, py), (gx, gy) in zip(pred_centers, gt_centers):
        ex = px - gx
        ey = py - gy
        errors_x.append(ex)
        errors_y.append(ey)
        errors_euc.append(np.sqrt(ex**2 + ey**2))
    
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    errors_euc = np.array(errors_euc)
    
    scale = working_dist / focal_length  # mm/px
    
    results = {
        'cx_rmse': np.sqrt(np.mean(errors_x**2)),
        'cy_rmse': np.sqrt(np.mean(errors_y**2)),
        'euc_rmse': np.sqrt(np.mean(errors_euc**2)),
        'cx_mae': np.mean(np.abs(errors_x)),
        'cy_mae': np.mean(np.abs(errors_y)),
        'euc_mae': np.mean(errors_euc),
        'sub5px_cx': 100 * np.mean(np.abs(errors_x) < 5),
        'sub5px_cy': 100 * np.mean(np.abs(errors_y) < 5),
        'sub5px_euc': 100 * np.mean(errors_euc < 5),
        'phys_cx_mm': np.sqrt(np.mean(errors_x**2)) * scale,
        'phys_cy_mm': np.sqrt(np.mean(errors_y**2)) * scale,
        'phys_euc_mm': np.sqrt(np.mean(errors_euc**2)) * scale,
        'median_euc': np.median(errors_euc),
        'n_detections': len(errors_euc),
        'scale_mm_per_px': scale,
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True, help='Path to CPL subset images')
    parser.add_argument('--gt-dir', type=str, required=True, help='Path to GT annotation dir')
    parser.add_argument('--focal-length', type=float, default=612.36)
    parser.add_argument('--working-dist', type=float, default=500)
    args = parser.parse_args()
    
    # Run inference, match with GT, compute metrics
    # (Implementation depends on your matching strategy)
    print("CPL Evaluation")
    print(f"Focal length: {args.focal_length} px")
    print(f"Working distance: {args.working_dist} mm")
    print(f"Scale: {args.working_dist/args.focal_length:.4f} mm/px")

if __name__ == '__main__':
    main()
