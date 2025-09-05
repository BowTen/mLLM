#!/usr/bin/env python3
"""
GEMM Performance Comparison Plot
Plots MyGEMM vs CUBLAS performance comparison from gemm_eval.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_gemm_performance(csv_path='gemm_eval.csv', output_path=None):
    """
    Plot GEMM performance comparison
    
    Args:
        csv_path: Path to the CSV file containing performance data
        output_path: Path to save the plot (optional)
    """
    
    # Read the CSV data
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get column names (excluding Size column for the data columns)
    data_columns = [col for col in df.columns if col != 'Size']
    
    # Plot both lines using original column names
    if len(data_columns) >= 2:
        plt.plot(df['Size'], df[data_columns[0]], 'o-', label=data_columns[0], linewidth=2, markersize=6)
        plt.plot(df['Size'], df[data_columns[1]], 's-', label=data_columns[1], linewidth=2, markersize=6)
    
    # Customize the plot
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('GEMM Performance Comparison: MyGEMM vs CUBLAS', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(200, max(df['Size']) + 200)
    max_val = max(df[data_columns[0]].max(), df[data_columns[1]].max()) if len(data_columns) >= 2 else df[data_columns[0]].max()
    plt.ylim(0, max_val * 1.1)
    
    # Add some statistics text
    if len(data_columns) >= 2:
        max_first = df[data_columns[0]].max()
        max_second = df[data_columns[1]].max()
        avg_ratio = (df[data_columns[0]] / df[data_columns[1]]).mean()
        
        stats_text = f'Max {data_columns[0]}: {max_first:.1f} GFLOPS\n'
        stats_text += f'Max {data_columns[1]}: {max_second:.1f} GFLOPS\n'
        stats_text += f'Avg {data_columns[0]}/{data_columns[1]}: {avg_ratio:.2f}'
    else:
        max_val = df[data_columns[0]].max()
        stats_text = f'Max {data_columns[0]}: {max_val:.1f} GFLOPS'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('gemm_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved to gemm_performance_comparison.png")
    
    plt.show()

def plot_performance_ratio(csv_path='gemm_eval.csv', output_path=None):
    """
    Plot the performance ratio (First column / Second column) vs matrix size
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    data_columns = [col for col in df.columns if col != 'Size']
    
    if len(data_columns) < 2:
        print("Error: Need at least 2 data columns for ratio calculation!")
        return
    
    df['Ratio'] = df[data_columns[0]] / df[data_columns[1]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Size'], df['Ratio'], 'ro-', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Performance')
    
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel(f'{data_columns[0]} / {data_columns[1]} Performance Ratio', fontsize=12)
    plt.title(f'GEMM Performance Ratio: {data_columns[0]} vs {data_columns[1]}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add average ratio line
    avg_ratio = df['Ratio'].mean()
    plt.axhline(y=avg_ratio, color='orange', linestyle=':', alpha=0.8, 
                label=f'Average Ratio: {avg_ratio:.2f}')
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        ratio_path = output_path.replace('.png', '_ratio.png')
        plt.savefig(ratio_path, dpi=300, bbox_inches='tight')
        print(f"Ratio plot saved to {ratio_path}")
    else:
        plt.savefig('gemm_performance_ratio.png', dpi=300, bbox_inches='tight')
        print("Ratio plot saved to gemm_performance_ratio.png")
    
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Plot GEMM performance comparison from CSV data')
    parser.add_argument('--csv', '-c', 
                       default='gemm_eval.csv',
                       help='Path to the CSV file containing performance data (default: gemm_eval.csv)')
    parser.add_argument('--output', '-o',
                       help='Output path for the plots (optional)')
    
    args = parser.parse_args()
    csv_file = args.csv
    output_path = args.output
    
    print(f"Using CSV file: {csv_file}")
    if output_path:
        print(f"Output path: {output_path}")
    
    print("Plotting GEMM performance comparison...")
    plot_gemm_performance(csv_file, output_path)
    
    print("\nPlotting performance ratio...")
    plot_performance_ratio(csv_file, output_path)
    
    # Print summary statistics
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        data_columns = [col for col in df.columns if col != 'Size']
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        if len(data_columns) >= 2:
            print(f"{data_columns[0]} Peak Performance: {df[data_columns[0]].max():.1f} GFLOPS")
            print(f"{data_columns[1]} Peak Performance: {df[data_columns[1]].max():.1f} GFLOPS")
            print(f"Average Performance Ratio: {(df[data_columns[0]] / df[data_columns[1]]).mean():.2f}")
            print(f"Best Performance Ratio: {(df[data_columns[0]] / df[data_columns[1]]).max():.2f}")
            
            # Find the size with best ratio
            best_ratio_idx = (df[data_columns[0]] / df[data_columns[1]]).idxmax()
            best_size = df.loc[best_ratio_idx, 'Size']
            best_ratio = df.loc[best_ratio_idx, data_columns[0]] / df.loc[best_ratio_idx, data_columns[1]]
            print(f"Best ratio at size {best_size}: {best_ratio:.2f}")
        elif len(data_columns) == 1:
            print(f"{data_columns[0]} Peak Performance: {df[data_columns[0]].max():.1f} GFLOPS")
        else:
            print("No data columns found!")

if __name__ == "__main__":
    main()
