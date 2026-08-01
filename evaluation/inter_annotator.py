"""Inter-annotator variability assessment for center-point annotations."""
import argparse
import numpy as np

def compute_iaa(a1_points, a2_points):
    """Compute inter-annotator agreement metrics."""
    diffs = a1_points - a2_points
    euc = np.sqrt(np.sum(diffs**2, axis=1))
    return {
        'cx_rmse': np.sqrt(np.mean(diffs[:, 0]**2)),
        'cy_rmse': np.sqrt(np.mean(diffs[:, 1]**2)),
        'euc_rmse': np.sqrt(np.mean(euc**2)),
        'cx_mae': np.mean(np.abs(diffs[:, 0])),
        'cy_mae': np.mean(np.abs(diffs[:, 1])),
        'euc_mae': np.mean(euc),
        'sub5px': 100 * np.mean(euc < 5),
        'n': len(euc),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a1', type=str, required=True, help='Annotator 1 keypoints (npy)')
    parser.add_argument('--a2', type=str, required=True, help='Annotator 2 keypoints (npy)')
    args = parser.parse_args()
    
    a1 = np.load(args.a1)  # (N, 2)
    a2 = np.load(args.a2)  # (N, 2)
    
    r = compute_iaa(a1, a2)
    print(f"Inter-annotator agreement (N={r['n']}):")
    print(f"  Euclidean RMSE: {r['euc_rmse']:.2f} px")
    print(f"  Euclidean MAE:  {r['euc_mae']:.2f} px")
    print(f"  <5px accuracy:  {r['sub5px']:.1f}%")

if __name__ == '__main__':
    main()
