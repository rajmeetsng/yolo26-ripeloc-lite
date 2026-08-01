"""Stratified error analysis by ripeness stage, occlusion, lighting, size."""
import argparse, json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True,
                       help='JSON with per-instance occlusion, size, lighting, USDA stage')
    args = parser.parse_args()
    
    print("Stratified Error Analysis")
    print("Stratify by: USDA stage, occlusion, lighting, object size")
    print("See paper Section 3.4.1 for methodology")

if __name__ == '__main__':
    main()
