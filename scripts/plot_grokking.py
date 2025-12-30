#!/usr/bin/env python
"""
Plot grokking curves from training logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_grokking(csv_path, output_path=None):
    """Plot train and validation accuracy/loss curves."""
    df = pd.read_csv(csv_path)

    # Use full_train_acc/loss (computed on full training set during validation)
    # These are in the same rows as val_accuracy/val_loss
    df_val = df[df['val_accuracy'].notna()].copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accuracy
    ax1 = axes[0]
    if len(df_val) > 0 and 'full_train_acc' in df_val.columns:
        ax1.plot(df_val['step'], df_val['full_train_acc'],
                 'r-', label='Train Accuracy', linewidth=2)
    if len(df_val) > 0:
        ax1.plot(df_val['step'], df_val['val_accuracy'],
                 'g-', label='Validation Accuracy', linewidth=2)

    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Generalization Beyond Overfitting')
    ax1.legend(loc='right')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Loss
    ax2 = axes[1]
    if len(df_val) > 0 and 'full_train_loss' in df_val.columns:
        ax2.plot(df_val['step'], df_val['full_train_loss'],
                 'r-', label='Train Loss', linewidth=2)
    if len(df_val) > 0:
        ax2.plot(df_val['step'], df_val['val_loss'],
                 'g-', label='Validation Loss', linewidth=2)

    # Add chance level loss line
    chance_loss = np.log(97)  # -log(1/97) for uniform distribution
    ax2.axhline(y=chance_loss, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance Level (ln(97) = {chance_loss:.2f})')

    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Train vs Validation Loss')
    ax2.legend(loc='upper right')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    # Print summary statistics
    if len(df_val) > 0:
        max_val_acc = df_val['val_accuracy'].max()
        max_val_acc_step = df_val.loc[df_val['val_accuracy'].idxmax(), 'step']
        print(f"\nSummary:")
        print(f"  Max validation accuracy: {max_val_acc:.2f}%")
        print(f"  Achieved at step: {int(max_val_acc_step)}")

        # Find when train accuracy first hit 99%
        if 'full_train_acc' in df_val.columns:
            train_99 = df_val[df_val['full_train_acc'] >= 99]
            if len(train_99) > 0:
                first_train_99 = train_99['step'].min()
                print(f"  Train accuracy first hit 99%: step {int(first_train_99)}")

        # Find when val accuracy first hit 99%
        val_99 = df_val[df_val['val_accuracy'] >= 99]
        if len(val_99) > 0:
            first_val_99 = val_99['step'].min()
            print(f"  Val accuracy first hit 99%: step {int(first_val_99)}")


def main():
    parser = argparse.ArgumentParser(description='Plot grokking curves')
    parser.add_argument('--csv', type=str,
                       default='lightning_logs/version_0/metrics.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (shows plot if not specified)')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return

    plot_grokking(args.csv, args.output)


if __name__ == "__main__":
    main()
