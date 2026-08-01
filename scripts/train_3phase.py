"""
YOLO26-RipeLoc Lite: 3-Phase Progressive Unfreezing Training
Usage: python scripts/train_3phase.py --data configs/tomato_dataset.yaml --cfg configs/yolo26n-ripeloc.yaml --seed 0
"""
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def train_phase(model, data, phase_cfg, phase_num, project, seed=0):
    """Train a single phase."""
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}: freeze={phase_cfg['freeze']}, lr={phase_cfg['lr0']}, epochs={phase_cfg['epochs']}")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=data,
        epochs=phase_cfg['epochs'],
        lr0=phase_cfg['lr0'],
        lrf=0.0001,
        optimizer=phase_cfg.get('optimizer', 'SGD'),
        momentum=phase_cfg.get('momentum', 0.937),
        weight_decay=phase_cfg.get('weight_decay', 0.0005),
        batch=16,
        imgsz=640,
        freeze=phase_cfg['freeze'],
        cos_lr=True,
        seed=seed,
        project=project,
        name=f'phase{phase_num}',
        # Augmentation
        hsv_h=0.042,
        hsv_s=0.5,
        hsv_v=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.2,
        erasing=0.1,
        scale=0.5,
    )
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='configs/tomato_dataset.yaml')
    parser.add_argument('--cfg', type=str, default='configs/yolo26n-ripeloc.yaml')
    parser.add_argument('--hyp', type=str, default='configs/training_hyperparams.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--weights', type=str, default='yolo26n.pt',
                       help='COCO pretrained weights')
    args = parser.parse_args()
    
    # Load hyperparameters
    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)
    
    # Phase 1: Frozen backbone
    model = YOLO(args.cfg).load(args.weights)
    train_phase(model, args.data, hyp['phase1'], 1, args.project, args.seed)
    
    # Phase 2: Partial unfreeze (layers 5-9)
    model = YOLO(f'{args.project}/phase1/weights/last.pt')
    train_phase(model, args.data, hyp['phase2'], 2, args.project, args.seed)
    
    # Phase 3: Full unfreeze
    model = YOLO(f'{args.project}/phase2/weights/last.pt')
    train_phase(model, args.data, hyp['phase3'], 3, args.project, args.seed)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best weights: {args.project}/phase3/weights/best.pt")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
