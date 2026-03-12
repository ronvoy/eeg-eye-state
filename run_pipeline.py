#!/usr/bin/env python3
"""
EEG Eye State Classification — Modular Pipeline Runner.

Usage:
    python run_pipeline.py --dataset data.csv --models rf,cnn --sections eda,ml

This is the modular entry point that uses the eeg_eye_analysis package.
For the full monolithic analysis, use: python script.py > report.md
"""
import argparse
import sys
import os

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="EEG Eye State Classification — Modular Pipeline")
    parser.add_argument("--dataset", type=str,
                        default="dataset/eeg_data_og.csv",
                        help="Path to CSV dataset")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated: rf,knn,svm,lr,gb,cnn1d,cnn2d,lstm")
    parser.add_argument("--sections", type=str, default="all",
                        help="Comma-separated: eda,preprocessing,features,dimreduction,ml,nn,comparison")
    parser.add_argument("--plot-dir", type=str, default="analysis-plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    # For now, delegate to the monolithic script
    # This will be replaced with modular calls as the package matures
    print(f"Dataset: {args.dataset}", file=sys.stderr)
    print(f"Models: {args.models}", file=sys.stderr)
    print(f"Sections: {args.sections}", file=sys.stderr)
    print(f"Plot dir: {args.plot_dir}", file=sys.stderr)

    # Import and run the full pipeline
    import script
    script.DATA_FILE = args.dataset
    script.PLOT_DIR = args.plot_dir
    os.makedirs(args.plot_dir, exist_ok=True)
    script.main()


if __name__ == "__main__":
    main()
