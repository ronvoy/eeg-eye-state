#!/usr/bin/env python3
"""Generate script.ipynb from script.py — run once then delete."""
import json

with open('script.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()  # list of strings, each with \n

total = len(lines)

def get(start, end):
    """Get lines[start-1:end-1] (1-indexed inclusive start, exclusive end)."""
    return lines[start - 1 : end - 1]

def trim_trailing(block):
    """Remove trailing blank/separator lines from a block."""
    while block and block[-1].strip() in ('', ):
        block.pop()
    while block and block[-1].strip().startswith('# ===='):
        block.pop()
    while block and block[-1].strip() in ('', ):
        block.pop()
    # also remove separator title lines like "# 2. Data Imputation"
    while block and block[-1].strip().startswith('# ') and not block[-1].startswith('    '):
        prev = block[-1].strip()
        if prev.startswith('# ====') or (len(prev) > 2 and prev[2:3].isdigit()):
            block.pop()
        else:
            break
    while block and block[-1].strip().startswith('# ===='):
        block.pop()
    while block and block[-1].strip() == '':
        block.pop()
    return block

def src(block):
    """Ensure block ends without trailing newline on last element."""
    if not block:
        return ['']
    result = list(block)
    if result[-1].endswith('\n'):
        result[-1] = result[-1].rstrip('\n')
    return result

def md(text):
    parts = text.split('\n')
    source = [p + '\n' for p in parts[:-1]] + [parts[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(block):
    return {
        "cell_type": "code", "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src(block)
    }

cells = []

# ========== Cell 1: Markdown Title ==========
cells.append(md(
    "# EEG Eye State Classification — Complete Analysis Pipeline\n\n"
    "**Dataset:** [UCI Machine Learning Repository — EEG Eye State]"
    "(https://archive.ics.uci.edu/dataset/264/eeg+eye+state)\n\n"
    "Notebook version of `script.py`. Each section is executed below with inline output.\n\n"
    "**Upgrades:** Bandpass filter (0.5–45 Hz), ICA artifact removal, "
    "frequency band power features, Pipeline-wrapped ML models, temporal EDA plots."
))

# ========== Cell 2: Imports (lines 1-67, modified) ==========
import_block = ['%matplotlib inline\n']
for line in get(1, 68):
    if 'matplotlib.use' in line:
        continue
    if 'import argparse' in line:
        continue
    import_block.append(line)
import_block.append('\n')
import_block.append('print("Imports loaded.")\n')
import_block.append('print(f"TensorFlow: {HAS_TF}")\n')
import_block.append('print(f"UMAP: {HAS_UMAP}")\n')
import_block.append('print(f"MNE: {HAS_MNE}")\n')
cells.append(code(import_block))

# ========== Cell 3: Configuration (lines 69-130, skip stdout reconfigure) ==========
config_block = []
skip_next = False
for line in get(69, 131):
    if 'hasattr(sys.stdout' in line:
        skip_next = True
        continue
    if skip_next:
        skip_next = False
        continue
    config_block.append(line)
cells.append(code(config_block))

# ========== Cell 4: Helper functions (lines 135-166, modified save_fig) ==========
helper_block = []
for line in get(135, 167):
    if line.strip() == 'plt.close("all")':
        helper_block.append('    plt.show()\n')
    helper_block.append(line)
cells.append(code(helper_block))

# ========== Section cells: define function + call ==========
sections = [
    # (title, func_start, func_end, execution_code)
    ("Table of Contents", 168, 248,
     "print_toc()"),

    ("1. Data Description", 248, 348,
     "df = pd.read_csv(DATA_FILE)\nprint(f'Loaded {len(df)} samples')\nsection_data_description(df)"),

    ("2. Data Imputation", 348, 375,
     "df = section_data_imputation(df)"),

    ("3. Data Visualization (Raw)", 375, 534,
     "section_data_viz_raw(df)"),

    ("4. Signal Preprocessing (Bandpass + ICA)", 534, 751,
     "df_raw_copy = df.copy()\ndf_clean = section_outlier_removal(df)"),

    ("5. Data Visualization (After Preprocessing)", 751, 794,
     "section_data_viz_cleaned(df_raw_copy, df_clean)"),

    ("6. Log-Normalization Assessment", 794, 905,
     "section_log_normalization(df_clean)"),

    ("7. Feature Engineering", 905, 1046,
     "df_eng, all_features = section_feature_engineering(df_clean)"),

    ("8. FFT, PSD & Spectrograms", 1046, 1166,
     "section_fft_psd_spectro(df_clean)"),

    ("9. Dimensionality Reduction", 1166, 1390,
     "section_dim_reduction(df_eng, all_features)"),

    ("10. ML Classification (Pipeline-based)", 1390, 1697,
     "ml_results = section_ml(df_eng, all_features)"),

    ("11. Neural Network Classification", 1697, 2200,
     "nn_results = section_neural_network(df_clean)"),

    ("12. Final Comparison", 2200, 2310,
     "section_final_comparison(ml_results, nn_results)"),
]

for title, fstart, fend, exec_code in sections:
    # Markdown header cell
    cells.append(md(f"## {title}"))

    # Code cell: function definition + execution
    func_block = list(get(fstart, fend))
    func_block = trim_trailing(func_block)
    func_block.append('\n')
    func_block.append('\n')
    func_block.append('# --- Execute ---\n')
    for exec_line in exec_code.split('\n'):
        func_block.append(exec_line + '\n')
    cells.append(code(func_block))

# ========== Build notebook ==========
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('script.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created script.ipynb with {len(cells)} cells")
for i, c in enumerate(cells):
    ctype = c['cell_type']
    nlines = len(c['source'])
    print(f"  Cell {i+1}: {ctype:8s} ({nlines} lines)")
