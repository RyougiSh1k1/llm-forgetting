"""
Simulate Figure 2(c)/(d) from "Understanding the Role of Training Regimes in Continual Learning"
This creates realistic plots demonstrating the theoretical relationship between sharpness and forgetting.

Based on Equation 5: F₁ ≈ (1/2) λ_max ||Δw||²
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def simulate_figure2c():
    """
    Simulate Figure 2(c): Forgetting vs (1/2) λ_max ||Δw||²

    This demonstrates the theoretical bound from the paper.
    """
    np.random.seed(42)

    # Simulate different training regimes for LoRA continual learning
    num_experiments = 15

    # Generate realistic values based on paper's observations
    # λ_max typically ranges from 0.01 to 0.1 for neural networks
    lambda_max_values = np.random.uniform(0.015, 0.08, num_experiments)

    # ||Δw|| displacement in LoRA subspace (smaller than full space)
    displacement_values = np.random.uniform(5.0, 25.0, num_experiments)

    # Compute sharpness measure: (1/2) λ_max ||Δw||²
    sharpness_measure = 0.5 * lambda_max_values * (displacement_values ** 2)

    # Actual forgetting follows the bound with some noise
    # F₁ ≈ (1/2) λ_max ||Δw||² + noise
    base_forgetting = sharpness_measure * 0.8  # Not always exact bound
    noise = np.random.normal(0, sharpness_measure * 0.15, num_experiments)
    actual_forgetting = np.maximum(0, base_forgetting + noise)

    # Create labels for different configurations
    labels = [
        "Dropout=0.5, LR=0.001", "Dropout=0.0, LR=0.01", "Dropout=0.3, LR=0.005",
        "BatchSize=16", "BatchSize=256", "BatchSize=64",
        "LR decay=0.9", "LR decay=1.0", "LR decay=0.5",
        "Task1", "Task2", "Task3",
        "LoRA r=16", "LoRA r=32", "LoRA r=64"
    ]

    # Create Figure 2(c) style plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    scatter = ax.scatter(sharpness_measure, actual_forgetting,
                        s=120, alpha=0.7, edgecolors='black', linewidth=1.5,
                        c=lambda_max_values, cmap='RdYlBu_r')

    # Add trend line
    z = np.polyfit(sharpness_measure, actual_forgetting, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sharpness_measure.min(), sharpness_measure.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
           label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

    # Compute correlation
    correlation = np.corrcoef(sharpness_measure, actual_forgetting)[0, 1]

    # Labels and formatting
    ax.set_xlabel(r'$\frac{1}{2} \lambda_{\mathrm{max}} \|\Delta w\|^2$',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Forgetting ($F_1$)', fontsize=14, fontweight='bold')
    ax.set_title('(c)', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=r'$\lambda_{\mathrm{max}}$')
    cbar.set_label(r'$\lambda_{\mathrm{max}}$ (Maximum Eigenvalue)', fontsize=11)

    # Annotate a few key points
    for i in [0, 5, 10]:
        ax.annotate(labels[i],
                   (sharpness_measure[i], actual_forgetting[i]),
                   fontsize=8, alpha=0.6,
                   xytext=(8, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    return fig, {
        'sharpness': sharpness_measure,
        'forgetting': actual_forgetting,
        'lambda_max': lambda_max_values,
        'displacement': displacement_values,
        'correlation': correlation
    }


def simulate_figure2d():
    """
    Simulate Figure 2(d): Eigenvalue spectrum comparison

    Shows how eigenvalue spectra differ across tasks and training regimes.
    """
    np.random.seed(42)

    # Number of eigenvalues to show
    num_eigenvalues = 20
    indices = np.arange(1, num_eigenvalues + 1)

    # Simulate eigenvalue spectra for different scenarios
    # Eigenvalues typically decay exponentially

    # Scenario 1: Task 1 (Wide/Flat minima - good for continual learning)
    base_wide = 0.025
    decay_wide = 0.85
    eigenvals_task1 = base_wide * (decay_wide ** np.arange(num_eigenvalues))
    eigenvals_task1 += np.random.normal(0, 0.001, num_eigenvalues)
    eigenvals_task1 = np.maximum(eigenvals_task1, 0)

    # Scenario 2: Task 2 (Sharp minima - prone to forgetting)
    base_sharp = 0.055
    decay_sharp = 0.75
    eigenvals_task2 = base_sharp * (decay_sharp ** np.arange(num_eigenvalues))
    eigenvals_task2 += np.random.normal(0, 0.002, num_eigenvalues)
    eigenvals_task2 = np.maximum(eigenvals_task2, 0)

    # Scenario 3: After Task 2 on Task 1 data (intermediate sharpness)
    base_medium = 0.038
    decay_medium = 0.82
    eigenvals_after = base_medium * (decay_medium ** np.arange(num_eigenvalues))
    eigenvals_after += np.random.normal(0, 0.0015, num_eigenvalues)
    eigenvals_after = np.maximum(eigenvals_after, 0)

    # Create Figure 2(d) style plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot eigenvalue spectra
    ax.plot(indices, eigenvals_task1, 'o-',
           label=f'Task 1 - ScienceQA (λ_max={eigenvals_task1[0]:.4f})',
           linewidth=2.5, markersize=7, color='#2E86AB', alpha=0.8)

    ax.plot(indices, eigenvals_task2, 's-',
           label=f'Task 2 - FOMC (λ_max={eigenvals_task2[0]:.4f})',
           linewidth=2.5, markersize=7, color='#A23B72', alpha=0.8)

    ax.plot(indices, eigenvals_after, '^-',
           label=f'After Task2 on Task1 (λ_max={eigenvals_after[0]:.4f})',
           linewidth=2.5, markersize=7, color='#F18F01', alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Eigenvalue Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Eigenvalue Magnitude', fontsize=14, fontweight='bold')
    ax.set_title("(d)", fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add shaded regions to highlight sharpness
    ax.axhspan(0.04, 0.06, alpha=0.1, color='red',
              label='Sharp region (high forgetting)')
    ax.axhspan(0.0, 0.03, alpha=0.1, color='green',
              label='Flat region (low forgetting)')

    # Add text annotation
    ax.text(0.98, 0.65,
           'Flatter spectrum\n→ More stable\n→ Less forgetting',
           transform=ax.transAxes,
           fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
           verticalalignment='top',
           horizontalalignment='right')

    ax.text(0.98, 0.35,
           'Sharper spectrum\n→ Less stable\n→ More forgetting',
           transform=ax.transAxes,
           fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3),
           verticalalignment='top',
           horizontalalignment='right')

    plt.tight_layout()

    return fig, {
        'task1': eigenvals_task1,
        'task2': eigenvals_task2,
        'after_task2': eigenvals_after
    }


def create_combined_figure():
    """Create combined figure with both (c) and (d) subplots."""
    fig = plt.figure(figsize=(16, 6))

    # Figure 2(c) - Left subplot
    ax1 = fig.add_subplot(121)

    np.random.seed(42)
    num_experiments = 15
    lambda_max_values = np.random.uniform(0.015, 0.08, num_experiments)
    displacement_values = np.random.uniform(5.0, 25.0, num_experiments)
    sharpness_measure = 0.5 * lambda_max_values * (displacement_values ** 2)
    base_forgetting = sharpness_measure * 0.8
    noise = np.random.normal(0, sharpness_measure * 0.15, num_experiments)
    actual_forgetting = np.maximum(0, base_forgetting + noise)

    scatter = ax1.scatter(sharpness_measure, actual_forgetting,
                         s=120, alpha=0.7, edgecolors='black', linewidth=1.5,
                         c=lambda_max_values, cmap='RdYlBu_r')

    z = np.polyfit(sharpness_measure, actual_forgetting, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sharpness_measure.min(), sharpness_measure.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    correlation = np.corrcoef(sharpness_measure, actual_forgetting)[0, 1]
    ax1.set_xlabel(r'$\frac{1}{2} \lambda_{\mathrm{max}} \|\Delta w\|^2$', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Forgetting ($F_1$)', fontsize=13, fontweight='bold')
    ax1.set_title('(c)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Figure 2(d) - Right subplot
    ax2 = fig.add_subplot(122)

    num_eigenvalues = 20
    indices = np.arange(1, num_eigenvalues + 1)

    eigenvals_task1 = 0.025 * (0.85 ** np.arange(num_eigenvalues))
    eigenvals_task2 = 0.055 * (0.75 ** np.arange(num_eigenvalues))
    eigenvals_after = 0.038 * (0.82 ** np.arange(num_eigenvalues))

    ax2.plot(indices, eigenvals_task1, 'o-', label='Task 1 (Flat)',
            linewidth=2.5, markersize=7, color='#2E86AB')
    ax2.plot(indices, eigenvals_task2, 's-', label='Task 2 (Sharp)',
            linewidth=2.5, markersize=7, color='#A23B72')
    ax2.plot(indices, eigenvals_after, '^-', label='After Task 2',
            linewidth=2.5, markersize=7, color='#F18F01')

    ax2.set_xlabel('Eigenvalue Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=13, fontweight='bold')
    ax2.set_title('(d)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def main():
    """Generate all simulation plots."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./simulated_figures_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SIMULATING FIGURE 2(c)/(d) FROM PAPER")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}\n")

    # Generate Figure 2(c)
    print("Generating Figure 2(c): Forgetting vs Sharpness...")
    fig_c, data_c = simulate_figure2c()
    path_c = os.path.join(output_dir, "figure2c_forgetting_vs_sharpness.png")
    fig_c.savefig(path_c, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {path_c}")
    print(f"  - Correlation: {data_c['correlation']:.3f}")
    print(f"  - λ_max range: [{data_c['lambda_max'].min():.4f}, {data_c['lambda_max'].max():.4f}]")
    print(f"  - Displacement range: [{data_c['displacement'].min():.2f}, {data_c['displacement'].max():.2f}]")

    # Generate Figure 2(d)
    print("\nGenerating Figure 2(d): Eigenvalue Spectrum...")
    fig_d, data_d = simulate_figure2d()
    path_d = os.path.join(output_dir, "figure2d_eigenvalue_spectrum.png")
    fig_d.savefig(path_d, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {path_d}")
    print(f"  - Task 1 λ_max: {data_d['task1'][0]:.4f} (flat)")
    print(f"  - Task 2 λ_max: {data_d['task2'][0]:.4f} (sharp)")
    print(f"  - After Task 2 λ_max: {data_d['after_task2'][0]:.4f} (medium)")

    # Generate combined figure
    print("\nGenerating combined Figure 2(c)+(d)...")
    fig_combined = create_combined_figure()
    path_combined = os.path.join(output_dir, "figure2_combined.png")
    fig_combined.savefig(path_combined, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {path_combined}")

    # Save data to file
    import json
    data_summary = {
        "figure_2c": {
            "correlation": float(data_c['correlation']),
            "lambda_max_range": [float(data_c['lambda_max'].min()),
                                float(data_c['lambda_max'].max())],
            "displacement_range": [float(data_c['displacement'].min()),
                                  float(data_c['displacement'].max())],
            "num_experiments": len(data_c['sharpness'])
        },
        "figure_2d": {
            "task1_lambda_max": float(data_d['task1'][0]),
            "task2_lambda_max": float(data_d['task2'][0]),
            "after_task2_lambda_max": float(data_d['after_task2'][0]),
            "num_eigenvalues": len(data_d['task1'])
        },
        "theoretical_relationship": "F1 ≈ (1/2) * lambda_max * ||Delta_w||^2",
        "paper_reference": "Mirzadeh et al. (2020) - Understanding the Role of Training Regimes in Continual Learning"
    }

    data_path = os.path.join(output_dir, "simulation_data.json")
    with open(data_path, 'w') as f:
        json.dump(data_summary, f, indent=2)
    print(f"  ✓ Data summary saved to: {data_path}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. {path_c}")
    print(f"  2. {path_d}")
    print(f"  3. {path_combined}")
    print(f"  4. {data_path}")
    print("\nThese figures demonstrate the theoretical relationship from the paper:")
    print("  • Wide/flat minima (low λ_max) → Less forgetting")
    print("  • Sharp minima (high λ_max) → More forgetting")
    print("  • Forgetting ≈ (1/2) λ_max ||Δw||²")
    print("\n" + "=" * 70 + "\n")

    plt.close('all')


if __name__ == "__main__":
    main()
