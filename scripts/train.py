"""Single-phase training script."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True, choices=[1,2,3])
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cfg', type=str, default='configs/yolo26n-ripeloc.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weights', type=str, default='yolo26n.pt')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    PHASES = {
        1: {'epochs': 50, 'lr0': 0.002, 'freeze': 10},
        2: {'epochs': 80, 'lr0': 0.001, 'freeze': 5},
        3: {'epochs': 120, 'lr0': 0.0003, 'freeze': 0},
    }
    cfg = PHASES[args.phase]
    
    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(args.cfg).load(args.weights)
    
    model.train(data=args.data, epochs=cfg['epochs'], lr0=cfg['lr0'],
                freeze=cfg['freeze'], batch=16, imgsz=640, seed=args.seed,
                optimizer='SGD', momentum=0.937, weight_decay=0.0005,
                cos_lr=True, hsv_h=0.042, hsv_s=0.5, hsv_v=0.5,
                fliplr=0.5, mosaic=1.0, mixup=0.3, copy_paste=0.2,
                erasing=0.1, scale=0.5,
                project='runs/train', name=f'phase{args.phase}')

if __name__ == '__main__':
    main()
