"""BN channel soft pruning + fine-tuning."""
import argparse, torch
import numpy as np
from ultralytics import YOLO

def collect_bn_gammas(model):
    gammas = []
    for m in model.model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            gammas.append(m.weight.data.abs().cpu().numpy())
    return np.concatenate(gammas)

def soft_prune(model, ratio=0.3):
    gammas = collect_bn_gammas(model)
    threshold = np.percentile(gammas, ratio * 100)
    pruned = 0
    total = 0
    for m in model.model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            mask = m.weight.data.abs() > threshold
            m.weight.data *= mask.float()
            m.bias.data *= mask.float()
            pruned += (~mask).sum().item()
            total += mask.numel()
    print(f"Soft pruned: {pruned}/{total} channels ({100*pruned/total:.1f}%)")
    print(f"Threshold: {threshold:.6f}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='configs/tomato_dataset.yaml')
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--finetune-lr', type=float, default=0.0005)
    args = parser.parse_args()
    
    model = YOLO(args.weights)
    print(f"Pre-prune params: {sum(p.numel() for p in model.model.parameters())/1e6:.2f}M")
    
    model = soft_prune(model, args.ratio)
    
    nonzero = sum((p != 0).sum().item() for p in model.model.parameters())
    print(f"Post-prune non-zero: {nonzero/1e6:.2f}M")
    
    if args.finetune_epochs > 0:
        print(f"\nFine-tuning for {args.finetune_epochs} epochs...")
        model.train(data=args.data, epochs=args.finetune_epochs,
                   lr0=args.finetune_lr, batch=16, imgsz=640,
                   project='runs/pruned', name='soft_pruned')
    
    torch.save(model.model.state_dict(), 'runs/pruned/soft_pruned.pt')
    print("Soft pruned model saved.")

if __name__ == '__main__':
    main()
