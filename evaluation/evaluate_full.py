"""Full evaluation pipeline: mAP + CPL + stratified errors."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='configs/tomato_dataset.yaml')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    model = YOLO(args.weights)
    
    # Standard evaluation
    results = model.val(data=args.data, split=args.split,
                       iou=0.45, conf=0.25, plots=True)
    
    print(f"\n{'='*50}")
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50:95:  {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print(f"{'='*50}")
    
    # Per-class
    for i, name in enumerate(results.names.values()):
        print(f"  {name}: AP@50={results.box.ap50[i]:.4f}, P={results.box.p[i]:.4f}, R={results.box.r[i]:.4f}")

if __name__ == '__main__':
    main()
