"""Bootstrap 95% confidence intervals for model comparison."""
import argparse, json
import numpy as np

def bootstrap_metric(values, n_bootstrap=1000, ci=95):
    """Compute bootstrap CI for a metric."""
    boot = [np.mean(np.random.choice(values, len(values), replace=True))
            for _ in range(n_bootstrap)]
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return np.mean(boot), lo, hi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True, help='JSON of per-image metrics')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    with open(args.predictions) as f:
        data = json.load(f)
    
    # data = [{'image': ..., 'ap50': ..., 'precision': ..., 'recall': ...}, ...]
    ap50s = [d['ap50'] for d in data]
    
    mean, lo, hi = bootstrap_metric(ap50s, args.n_bootstrap)
    print(f"mAP@50: {mean:.2f} [{lo:.2f}, {hi:.2f}]")

if __name__ == '__main__':
    main()
