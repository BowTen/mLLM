#!/usr/bin/env python3
"""
Token Speed Performance Plot
Plots token generation speed vs sequence length from token_speed.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_token_speed(csv_path='token_speed.csv', output_path=None):
    """
    Plot token generation speed vs sequence length
    
    Args:
        csv_path: Path to the CSV file containing token speed data
        output_path: Path to save the plot (optional)
    """
    
    # Read the CSV data
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Check if data exists
    if df.empty or len(df) == 0:
        print(f"Error: CSV file {csv_path} is empty or contains no data!")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot token speed
    plt.plot(df['Sequence Length'], df['Tokens per Second'], 'o-', 
             color='#2E86AB', linewidth=2, markersize=6, label='Token Generation Speed')
    
    # Customize the plot
    plt.xlabel('Sequence Length (tokens)', fontsize=12)
    plt.ylabel('Tokens per Second', fontsize=12)
    plt.title('Token Generation Speed vs Sequence Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if len(df) > 0:
        plt.xlim(0, max(df['Sequence Length']) * 1.05)
        plt.ylim(0, max(df['Tokens per Second']) * 1.1)
    
    # Add some statistics text
    if len(df) > 0:
        max_speed = df['Tokens per Second'].max()
        min_speed = df['Tokens per Second'].min()
        avg_speed = df['Tokens per Second'].mean()
        
        stats_text = f'Max Speed: {max_speed:.2f} tokens/sec\n'
        stats_text += f'Min Speed: {min_speed:.2f} tokens/sec\n'
        stats_text += f'Avg Speed: {avg_speed:.2f} tokens/sec'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('token_speed_performance.png', dpi=300, bbox_inches='tight')
        print("Plot saved to token_speed_performance.png")
    
    plt.show()

def plot_throughput_trend(csv_path='token_speed.csv', output_path=None):
    """
    Plot throughput trend with moving average
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty or len(df) == 0:
        print(f"Error: CSV file {csv_path} is empty or contains no data!")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    plt.plot(df['Sequence Length'], df['Tokens per Second'], 'o-', 
             alpha=0.7, linewidth=1, markersize=4, label='Raw Data', color='lightblue')
    
    # Add moving average if we have enough data points
    if len(df) >= 5:
        window_size = min(5, len(df) // 3)
        moving_avg = df['Tokens per Second'].rolling(window=window_size, center=True).mean()
        plt.plot(df['Sequence Length'], moving_avg, '-', 
                 linewidth=3, label=f'Moving Average (window={window_size})', color='red')
    
    plt.xlabel('Sequence Length (tokens)', fontsize=12)
    plt.ylabel('Tokens per Second', fontsize=12)
    plt.title('Token Generation Throughput Trend', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add trend analysis
    if len(df) >= 3:
        # Simple linear trend
        z = np.polyfit(df['Sequence Length'], df['Tokens per Second'], 1)
        p = np.poly1d(z)
        plt.plot(df['Sequence Length'], p(df['Sequence Length']), '--', 
                 alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.4f})', color='orange')
        plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        trend_path = output_path.replace('.png', '_trend.png')
        plt.savefig(trend_path, dpi=300, bbox_inches='tight')
        print(f"Trend plot saved to {trend_path}")
    else:
        plt.savefig('token_speed_trend.png', dpi=300, bbox_inches='tight')
        print("Trend plot saved to token_speed_trend.png")
    
    plt.show()

def plot_efficiency_analysis(csv_path='token_speed.csv', output_path=None):
    """
    Plot efficiency analysis showing speed degradation with sequence length
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty or len(df) == 0:
        print(f"Error: CSV file {csv_path} is empty or contains no data!")
        return
    
    # Calculate efficiency relative to first measurement
    if len(df) > 1:
        baseline_speed = df['Tokens per Second'].iloc[0]
        df['Efficiency %'] = (df['Tokens per Second'] / baseline_speed) * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Sequence Length'], df['Efficiency %'], 's-', 
                 color='green', linewidth=2, markersize=6, label='Efficiency %')
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Baseline (100%)')
        
        plt.xlabel('Sequence Length (tokens)', fontsize=12)
        plt.ylabel('Efficiency (%)', fontsize=12)
        plt.title('Token Generation Efficiency vs Sequence Length', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add efficiency statistics
        min_eff = df['Efficiency %'].min()
        avg_eff = df['Efficiency %'].mean()
        
        stats_text = f'Min Efficiency: {min_eff:.1f}%\n'
        stats_text += f'Avg Efficiency: {avg_eff:.1f}%\n'
        stats_text += f'Baseline Speed: {baseline_speed:.2f} tokens/sec'
        
        plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            eff_path = output_path.replace('.png', '_efficiency.png')
            plt.savefig(eff_path, dpi=300, bbox_inches='tight')
            print(f"Efficiency plot saved to {eff_path}")
        else:
            plt.savefig('token_speed_efficiency.png', dpi=300, bbox_inches='tight')
            print("Efficiency plot saved to token_speed_efficiency.png")
        
        plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Plot token generation speed from CSV data')
    parser.add_argument('--csv', '-c', 
                       default='token_speed.csv',
                       help='Path to the CSV file containing token speed data (default: token_speed.csv)')
    parser.add_argument('--output', '-o',
                       help='Output path for the plots (optional)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate all types of plots (speed, trend, efficiency)')
    
    args = parser.parse_args()
    csv_file = args.csv
    output_path = args.output
    
    print(f"Using CSV file: {csv_file}")
    if output_path:
        print(f"Output path: {output_path}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        print("Please run the token_speed_cal program first to generate the data.")
        return
    
    # Read and validate data
    try:
        df = pd.read_csv(csv_file)
        if df.empty or len(df) == 0:
            print(f"Warning: CSV file {csv_file} is empty or contains no data!")
            print("Please run the token_speed_cal program to generate data.")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Found {len(df)} data points in the CSV file")
    
    if args.all:
        print("Generating all plots...")
        print("\n1. Plotting token speed performance...")
        plot_token_speed(csv_file, output_path)
        
        print("\n2. Plotting throughput trend...")
        plot_throughput_trend(csv_file, output_path)
        
        print("\n3. Plotting efficiency analysis...")
        plot_efficiency_analysis(csv_file, output_path)
    else:
        print("Plotting token speed performance...")
        plot_token_speed(csv_file, output_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TOKEN SPEED SUMMARY")
    print("="*50)
    
    print(f"Total measurements: {len(df)}")
    print(f"Sequence length range: {df['Sequence Length'].min()} - {df['Sequence Length'].max()} tokens")
    print(f"Peak speed: {df['Tokens per Second'].max():.2f} tokens/sec")
    print(f"Minimum speed: {df['Tokens per Second'].min():.2f} tokens/sec")
    print(f"Average speed: {df['Tokens per Second'].mean():.2f} tokens/sec")
    print(f"Speed variance: {df['Tokens per Second'].var():.2f}")
    
    # Performance degradation analysis
    if len(df) > 1:
        initial_speed = df['Tokens per Second'].iloc[0]
        final_speed = df['Tokens per Second'].iloc[-1]
        degradation = ((initial_speed - final_speed) / initial_speed) * 100
        print(f"Speed degradation: {degradation:.1f}%")

if __name__ == "__main__":
    main()
