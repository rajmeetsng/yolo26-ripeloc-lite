"""Inference with CPL center-point extraction."""
import argparse, json
from pathlib import Path
from ultralytics import YOLO
from models.cdh import CPLModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--save-cpl', action='store_true')
    parser.add_argument('--save-dir', type=str, default='runs/detect')
    args = parser.parse_args()
    
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf,
                           iou=args.iou, save=True, project=args.save_dir)
    
    cpl = CPLModule()
    all_cpl = []
    
    for r in results:
        if r.boxes is None:
            continue
        dets = r.boxes.data  # (N, 6): x1,y1,x2,y2,conf,cls
        centers, ripe_mask = cpl.extract_center_points(dets, refine=True)
        
        for j, (cx, cy) in enumerate(centers):
            det_idx = ripe_mask.nonzero()[j].item()
            all_cpl.append({
                'image': str(r.path),
                'cx': round(cx.item(), 2),
                'cy': round(cy.item(), 2),
                'confidence': round(dets[det_idx, 4].item(), 4),
                'bbox': dets[det_idx, :4].tolist(),
            })
    
    if args.save_cpl:
        out = Path(args.save_dir) / 'cpl_results.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(all_cpl, f, indent=2)
        print(f"CPL results saved: {out} ({len(all_cpl)} ripe detections)")

if __name__ == '__main__':
    main()
