#!/usr/bin/env python3
"""
fill_na.py

Usage:
    python fill_na.py --input eeg_data.csv
    python fill_na.py --input eeg_data.csv --output eeg_data_filled.csv
    python fill_na.py --input eeg_data.csv --overwrite   # will back up original file

Requires:
    pip install pandas
"""

import argparse
import os
import shutil
import pandas as pd
import numpy as np
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(description="Find missing values in a CSV and fill them with the string 'na'.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=False, help="Output CSV file path (default: input_filled.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input file (a backup will be created: input.bak.csv)")
    args = parser.parse_args(argv)

    infile = args.input
    if not os.path.isfile(infile):
        print(f"Error: input file not found: {infile}", file=sys.stderr)
        sys.exit(2)

    # Determine output path
    if args.overwrite:
        backup_path = infile + ".bak.csv"
        shutil.copy2(infile, backup_path)
        outfile = infile
        print(f"Backup of original created: {backup_path}")
    else:
        if args.output:
            outfile = args.output
        else:
            base, ext = os.path.splitext(infile)
            outfile = f"{base}_filled{ext}"

    # Read CSV. Use default NA detection and also mark purely-whitespace strings as NaN.
    df = pd.read_csv(infile, keep_default_na=True, na_values=None, dtype=object)  # read everything as object to avoid dtype surprises

    # Replace any cell that is only whitespace (including empty string) with actual NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Show missing counts before
    missing_before = df.isna().sum()
    total_missing_before = missing_before.sum()
    print("Missing values per column (before):")
    print(missing_before.to_string())
    print(f"Total missing before: {total_missing_before}")

    # Fill NaN with 'na' (string)
    df_filled = df.fillna('na')

    # Show missing counts after
    missing_after = df_filled.isna().sum()
    print("\nMissing values per column (after):")
    print(missing_after.to_string())
    print(f"Total missing after: {missing_after.sum()}")

    # Save output CSV
    df_filled.to_csv(outfile, index=False)
    print(f"\nFilled CSV written to: {outfile}")

if __name__ == "__main__":
    main()