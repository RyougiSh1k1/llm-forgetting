"""Generate combined plots from all experiment results."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def collect_all_experiments(output_dir="./outputs"):
    """
    Collect results from all experiments.

    Args:
        output_dir: Base output directory

    Returns:
        List of dictionaries with experiment data
    """
    all_data = []

    # Find all results.json files
    results_files = glob(os.path.join(output_dir, "experiment_*/results.json"))

    print(f"Found {len(results_files)} experiments\n")

    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract key metrics
            data_point = {
                'experiment_name': os.path.basename(os.path.dirname(results_file)),
                'forgetting': results['forgetting_metrics']['absolute_forgetting'],
                'relative_forgetting_pct': results['forgetting_metrics']['relative_forgetting_pct'],
                'lambda_max': results['landscape_task1_after_task2']['lambda_max_lora'],
                'displacement': results['landscape_task1_after_task2']['lora_displacement'],
                'sharpness': 0.5 * results['landscape_task1_after_task2']['lambda_max_lora'] *
                            (results['landscape_task1_after_task2']['lora_displacement'] ** 2),
                'task1_acc_before': results['forgetting_metrics']['task1_acc_before'],
                'task1_acc_after': results['forgetting_metrics']['task1_acc_after'],
                'task2_acc_after': results['forgetting_metrics']['task2_acc_after'],
                'config': results['config']
            }
            all_data.append(data_point)
            print(f"✓ {data_point['experiment_name']}: "
                  f"forgetting={data_point['forgetting']:.3f}, "
                  f"sharpness={data_point['sharpness']:.1f}")

        except Exception as e:
            print(f"✗ Error loading {results_file}: {e}")

    return all_data

def plot_forgetting_vs_sharpness_combined(all_data, output_dir="./outputs"):
    """
    Create combined forgetting vs sharpness plot with all experiments.

    Args:
        all_data: List of experiment data dictionaries
        output_dir: Output directory
    """
    if len(all_data) == 0:
        print("No data to plot!")
        return

    # Extract arrays
    sharpness = np.array([d['sharpness'] for d in all_data])
    forgetting = np.array([d['forgetting'] for d in all_data])
    labels = [d['experiment_name'].replace('experiment_', '') for d in all_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    scatter = ax.scatter(sharpness, forgetting, s=150, alpha=0.7,
                        edgecolors='k', linewidth=1.5, c=forgetting,
                        cmap='RdYlGn_r', vmin=0, vmax=max(forgetting)*1.1)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Forgetting', rotation=270, labelpad=20, fontsize=12)

    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(label, (sharpness[i], forgetting[i]),
                   fontsize=7, alpha=0.7, xytext=(5, 5),
                   textcoords='offset points')

    # Fit trend line
    if len(sharpness) > 1:
        z = np.polyfit(sharpness, forgetting, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sharpness.min(), sharpness.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2,
               label=f'Linear fit: y={z[0]:.2e}x+{z[1]:.3f}')

        # Compute correlation
        corr = np.corrcoef(sharpness, forgetting)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=10)

    ax.set_xlabel(r'Sharpness: $\frac{1}{2} \lambda_{max} \|\Delta w\|^2$', fontsize=14)
    ax.set_ylabel('Forgetting (Absolute)', fontsize=14)
    ax.set_title(f'Forgetting vs Sharpness Analysis ({len(all_data)} experiments)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_forgetting_vs_sharpness.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Combined plot saved to: {plot_path}")

    # Also save high-res PDF
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(sharpness, forgetting, s=150, alpha=0.7,
                        edgecolors='k', linewidth=1.5, c=forgetting,
                        cmap='RdYlGn_r', vmin=0, vmax=max(forgetting)*1.1)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Forgetting', rotation=270, labelpad=20, fontsize=12)

    if len(sharpness) > 1:
        z = np.polyfit(sharpness, forgetting, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sharpness.min(), sharpness.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2,
               label=f'Linear fit: y={z[0]:.2e}x+{z[1]:.3f}')
        corr = np.corrcoef(sharpness, forgetting)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=10)

    ax.set_xlabel(r'Sharpness: $\frac{1}{2} \lambda_{max} \|\Delta w\|^2$', fontsize=14)
    ax.set_ylabel('Forgetting (Absolute)', fontsize=14)
    ax.set_title(f'Forgetting vs Sharpness Analysis ({len(all_data)} experiments)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path_pdf = os.path.join(output_dir, "combined_forgetting_vs_sharpness.pdf")
    plt.savefig(plot_path_pdf, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ PDF version saved to: {plot_path_pdf}")

def print_summary_statistics(all_data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    forgetting_vals = np.array([d['forgetting'] for d in all_data])
    sharpness_vals = np.array([d['sharpness'] for d in all_data])

    print(f"\nNumber of experiments: {len(all_data)}")
    print(f"\nForgetting:")
    print(f"  Mean: {np.mean(forgetting_vals):.4f}")
    print(f"  Std:  {np.std(forgetting_vals):.4f}")
    print(f"  Min:  {np.min(forgetting_vals):.4f}")
    print(f"  Max:  {np.max(forgetting_vals):.4f}")

    print(f"\nSharpness:")
    print(f"  Mean: {np.mean(sharpness_vals):.2f}")
    print(f"  Std:  {np.std(sharpness_vals):.2f}")
    print(f"  Min:  {np.min(sharpness_vals):.2f}")
    print(f"  Max:  {np.max(sharpness_vals):.2f}")

    if len(forgetting_vals) > 1:
        corr = np.corrcoef(sharpness_vals, forgetting_vals)[0, 1]
        print(f"\nCorrelation (Sharpness vs Forgetting): {corr:.4f}")

def main():
    """Main function."""
    print("="*70)
    print("GENERATING COMBINED PLOTS FROM ALL EXPERIMENTS")
    print("="*70 + "\n")

    # Collect all experiments
    all_data = collect_all_experiments()

    if len(all_data) == 0:
        print("\nNo experiment results found!")
        return

    # Generate combined plot
    plot_forgetting_vs_sharpness_combined(all_data)

    # Print statistics
    print_summary_statistics(all_data)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
