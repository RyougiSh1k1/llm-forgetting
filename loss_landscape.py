"""Loss landscape analysis module for examining flatness and catastrophic forgetting."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
from tqdm import tqdm


def generate_random_direction(model, seed: int = 42, device: str = 'cuda'):
    """
    Generate a random direction in parameter space with same dimension as model parameters.

    Args:
        model: Model
        seed: Random seed
        device: Device to place tensors on

    Returns:
        Random direction tensor (normalized)
    """
    torch.manual_seed(seed)
    direction = []

    for param in model.parameters():
        if param.requires_grad:
            random_dir = torch.randn_like(param.data)
            direction.append(random_dir.view(-1).to(device))

    direction = torch.cat(direction)
    # Normalize direction
    direction = direction / torch.norm(direction)

    return direction


def set_parameters_along_direction(model, original_params, direction, alpha: float):
    """
    Set model parameters along a direction: theta = theta_0 + alpha * direction.

    Args:
        model: Model
        original_params: Original parameter vector
        direction: Direction vector
        alpha: Step size
    """
    new_params = original_params + alpha * direction

    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data = new_params[offset:offset + numel].view(param.shape).to(param.device)
            offset += numel


def compute_loss(model, dataloader, device):
    """
    Compute average loss on dataset.

    Args:
        model: Model
        dataloader: DataLoader
        device: Device

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_2d_loss_landscape(
    model,
    dataloader,
    device,
    center_params,
    dir1,
    dir2,
    distance: float = 1.0,
    num_points: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D loss landscape around a point in parameter space.

    Args:
        model: Model
        dataloader: DataLoader for computing loss
        device: Device
        center_params: Center point parameters (1D tensor)
        dir1: First direction (normalized)
        dir2: Second direction (normalized)
        distance: Maximum distance to explore
        num_points: Number of points per dimension

    Returns:
        Tuple of (alphas, betas, loss_grid) where loss_grid[i,j] is loss at (alphas[i], betas[j])
    """
    alphas = np.linspace(-distance, distance, num_points)
    betas = np.linspace(-distance, distance, num_points)

    loss_grid = np.zeros((num_points, num_points))

    print(f"Computing 2D loss landscape ({num_points}x{num_points} grid)...")

    for i, alpha in enumerate(tqdm(alphas)):
        for j, beta in enumerate(betas):
            # Set parameters: theta = center + alpha*dir1 + beta*dir2
            new_params = center_params + alpha * dir1 + beta * dir2
            offset = 0
            for param in model.parameters():
                if param.requires_grad:
                    numel = param.numel()
                    param.data = new_params[offset:offset + numel].view(param.shape).to(param.device)
                    offset += numel

            # Compute loss
            loss = compute_loss(model, dataloader, device)
            loss_grid[i, j] = loss

            # Clear CUDA cache to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Restore center parameters
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data = center_params[offset:offset + numel].view(param.shape).to(param.device)
            offset += numel

    return alphas, betas, loss_grid


def plot_loss_landscape(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    title: str = "Loss Landscape",
    save_path: str = None
):
    """
    Plot 2D loss landscape.

    Args:
        alphas: Alpha values
        betas: Beta values
        loss_grid: Loss values grid
        title: Plot title
        save_path: Path to save plot
    """
    fig = plt.figure(figsize=(12, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax1.plot_surface(X, Y, loss_grid.T, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'{title} (3D)')

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, loss_grid.T, levels=20, cmap='viridis')
    ax2.set_xlabel('Direction 1')
    ax2.set_ylabel('Direction 2')
    ax2.set_title(f'{title} (Contour)')
    plt.colorbar(contour, ax=ax2, label='Loss')

    # Mark center
    ax2.plot(0, 0, 'r*', markersize=15, label='Center')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def compute_sharpness_metrics(loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> Dict[str, float]:
    """
    Compute sharpness metrics from loss landscape.

    Args:
        loss_grid: Loss values grid
        alphas: Alpha values
        betas: Beta values

    Returns:
        Dictionary of sharpness metrics
    """
    center_idx = len(alphas) // 2
    center_loss = loss_grid[center_idx, center_idx]

    # Maximum loss in neighborhood
    max_loss = np.max(loss_grid)

    # Average loss in neighborhood
    avg_loss = np.mean(loss_grid)

    # Standard deviation (measure of variation)
    std_loss = np.std(loss_grid)

    # Maximum curvature (approximation)
    # Compute second derivatives
    dx = alphas[1] - alphas[0]
    dy = betas[1] - betas[0]

    d2_dx2 = np.gradient(np.gradient(loss_grid, axis=0), axis=0) / (dx ** 2)
    d2_dy2 = np.gradient(np.gradient(loss_grid, axis=1), axis=1) / (dy ** 2)

    max_curvature = np.max(np.abs(d2_dx2) + np.abs(d2_dy2))

    return {
        "center_loss": center_loss,
        "max_loss": max_loss,
        "avg_loss": avg_loss,
        "std_loss": std_loss,
        "max_curvature": max_curvature,
        "loss_range": max_loss - center_loss
    }


def get_lora_parameters(model, device) -> torch.Tensor:
    """
    Get only LoRA parameters as a single vector (for LoRA subspace analysis).

    Args:
        model: PEFT model with LoRA
        device: Device

    Returns:
        Flattened LoRA parameter vector
    """
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_params.append(param.data.view(-1).to(device))

    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found! Make sure the model has LoRA adapters.")

    return torch.cat(lora_params)


def compute_hessian_eigenvalues_lora_subspace(model, dataloader, device, num_eigenvalues: int = 20) -> np.ndarray:
    """
    Compute top eigenvalues of the Hessian in LoRA subspace.

    Following "Understanding the Role of Training Regimes in Continual Learning" (Mirzadeh et al. 2020),
    this computes the Hessian with respect to LoRA parameters only, which is much more efficient
    and relevant for LoRA-based continual learning.

    The forgetting bound is: F1 ≈ (1/2) λ_max ||Δw||²
    where λ_max is the maximum Hessian eigenvalue in LoRA subspace.

    Args:
        model: PEFT model with LoRA
        dataloader: DataLoader for computing loss
        device: Device
        num_eigenvalues: Number of top eigenvalues to compute

    Returns:
        Array of top eigenvalues (sorted descending)
    """
    try:
        print(f"    Computing Hessian eigenvalues in LoRA subspace...")
        model.eval()

        # Collect LoRA parameters only
        lora_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                lora_params.append(param)

        if len(lora_params) == 0:
            print("    Warning: No LoRA parameters found, falling back to all trainable params")
            lora_params = [p for p in model.parameters() if p.requires_grad]

        print(f"    LoRA subspace dimension: {sum(p.numel() for p in lora_params)}")

        # Get a batch for computing Hessian
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Compute loss and gradients
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Get gradient w.r.t. LoRA parameters
        grads = torch.autograd.grad(loss, lora_params, create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grads if g is not None])

        if grad_vec.numel() == 0 or torch.all(torch.abs(grad_vec) < 1e-10):
            print("    Warning: Gradient is near zero, using approximation")
            return np.zeros(num_eigenvalues)

        eigenvalues = []

        # Power iteration for top eigenvalues
        print(f"    Running power iteration for top {num_eigenvalues} eigenvalues...")
        for i in range(min(num_eigenvalues, 10)):  # Limit iterations
            # Random initialization
            v = torch.randn(grad_vec.size(0), device=device, dtype=grad_vec.dtype)
            v = v / (torch.norm(v) + 1e-10)

            # Power iteration
            for iter_idx in range(20):  # Increased iterations for convergence
                # Compute Hessian-vector product: Hv = ∇(g^T v)
                Hv_list = torch.autograd.grad(
                    grad_vec, lora_params, grad_outputs=v,
                    retain_graph=(i < min(num_eigenvalues, 10) - 1),
                    allow_unused=True
                )
                Hv = torch.cat([h.reshape(-1) if h is not None else torch.zeros_like(p).reshape(-1)
                               for h, p in zip(Hv_list, lora_params)])

                # Normalize
                norm = torch.norm(Hv)
                if norm > 1e-10:
                    v = Hv / norm
                else:
                    break

            # Compute Rayleigh quotient for eigenvalue
            if norm > 1e-10:
                eigenvalue = torch.dot(v, Hv).item()
                eigenvalues.append(max(0.0, eigenvalue))  # Ensure non-negative
            else:
                eigenvalues.append(0.0)

        # Pad with zeros if needed
        while len(eigenvalues) < num_eigenvalues:
            eigenvalues.append(0.0)

        # Sort in descending order
        eigenvalues = sorted(eigenvalues, reverse=True)

        return np.array(eigenvalues[:num_eigenvalues])

    except Exception as e:
        print(f"    Warning: Could not compute LoRA Hessian eigenvalues: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(num_eigenvalues)


def compute_hessian_eigenvalues(model, dataloader, device, num_eigenvalues: int = 20) -> np.ndarray:
    """
    Compute top eigenvalues of the Hessian using power iteration.

    This approximates the Hessian eigenvalues at a trained checkpoint.
    Based on the methods in Mirzadeh et al. 2020.

    Args:
        model: Model
        dataloader: DataLoader for computing loss
        device: Device
        num_eigenvalues: Number of top eigenvalues to compute

    Returns:
        Array of top eigenvalues (sorted descending)
    """
    try:
        print(f"    Computing Hessian eigenvalues (this may take a minute)...")
        model.eval()

        # Use PyTorch Hessian eigenvalue library if available
        try:
            from pytorch_hessian_eigenthings import compute_hessian_eigenthings

            # Collect multiple batches for better approximation
            num_batches = min(5, len(dataloader))

            def loss_fn(model_params, batch):
                # Temporarily set model params
                offset = 0
                for p in model.parameters():
                    if p.requires_grad:
                        numel = p.numel()
                        p.data = model_params[offset:offset+numel].view(p.shape)
                        offset += numel

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                return outputs.loss

            eigenvals, _ = compute_hessian_eigenthings(
                model,
                dataloader,
                loss_fn,
                num_eigenthings=num_eigenvalues,
                mode='power_iter',
                max_samples=num_batches
            )

            return np.array(eigenvals[:num_eigenvalues])

        except ImportError:
            # Fallback: Use simpler approximation with finite differences
            print("    Using finite difference approximation (install pytorch-hessian-eigenthings for better results)")

            params = [p for p in model.parameters() if p.requires_grad]

            # Get loss at current point
            batch = next(iter(dataloader))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Compute loss and gradients
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Get gradient
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = torch.cat([g.reshape(-1) for g in grads if g is not None])

            if grad_vec.numel() == 0 or torch.all(grad_vec == 0):
                print("    Warning: Gradient is zero or empty, returning zero eigenvalues")
                return np.zeros(num_eigenvalues)

            eigenvalues = []

            # Simplified power iteration
            for i in range(min(num_eigenvalues, 5)):  # Limit to avoid timeout
                # Random direction
                v = torch.randn_like(grad_vec)
                v = v / (torch.norm(v) + 1e-10)

                # Power iteration (fewer iterations)
                for iter_idx in range(5):
                    # Compute Hessian-vector product via double backward
                    Hv_list = torch.autograd.grad(
                        grad_vec, params, grad_outputs=v,
                        retain_graph=(i < num_eigenvalues - 1),
                        allow_unused=True
                    )
                    Hv = torch.cat([h.reshape(-1) if h is not None else torch.zeros_like(p).reshape(-1)
                                   for h, p in zip(Hv_list, params)])

                    # Normalize
                    norm = torch.norm(Hv)
                    if norm > 1e-10:
                        v = Hv / norm
                    else:
                        break

                # Compute Rayleigh quotient
                eigenvalue = (v @ Hv).item() if norm > 1e-10 else 0.0
                eigenvalues.append(max(0.0, eigenvalue))  # Ensure non-negative

            # Pad with zeros if needed
            while len(eigenvalues) < num_eigenvalues:
                eigenvalues.append(0.0)

            return np.array(eigenvalues)

    except Exception as e:
        print(f"    Warning: Could not compute Hessian eigenvalues: {e}")
        print(f"    Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("    Returning zeros as placeholder")
        return np.zeros(num_eigenvalues)


def get_model_parameters(model, device) -> torch.Tensor:
    """
    Get all trainable parameters as a single vector.

    Args:
        model: Model
        device: Device

    Returns:
        Flattened parameter vector
    """
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.data.view(-1).to(device))
    return torch.cat(params)


def compute_parameter_distance(params1: torch.Tensor, params2: torch.Tensor) -> float:
    """
    Compute L2 distance between two parameter vectors.

    Args:
        params1: First parameter vector
        params2: Second parameter vector

    Returns:
        L2 distance
    """
    return torch.norm(params1 - params2).item()


def plot_forgetting_vs_sharpness(
    results_dict: Dict,
    output_dir: str,
    filename: str = "forgetting_vs_sharpness.png"
):
    """
    Create Figure 2(c)/(d) style plots from the paper:
    - (c) Forgetting vs λ_max * ||Δw||²
    - (d) Additional analysis plots

    Args:
        results_dict: Dictionary containing results from multiple experiments
                     Format: {experiment_name: {
                         'lambda_max': float,
                         'displacement': float,
                         'forgetting': float,
                         'task_name': str
                     }}
        output_dir: Output directory
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    lambda_max_list = []
    displacement_list = []
    forgetting_list = []
    labels = []

    for exp_name, data in results_dict.items():
        if all(k in data for k in ['lambda_max', 'displacement', 'forgetting']):
            lambda_max_list.append(data['lambda_max'])
            displacement_list.append(data['displacement'])
            forgetting_list.append(data['forgetting'])
            labels.append(data.get('task_name', exp_name))

    lambda_max_arr = np.array(lambda_max_list)
    displacement_arr = np.array(displacement_list)
    forgetting_arr = np.array(forgetting_list)

    # Plot (c): Forgetting vs λ_max * ||Δw||²
    sharpness_measure = 0.5 * lambda_max_arr * (displacement_arr ** 2)

    ax1 = axes[0]
    ax1.scatter(sharpness_measure, forgetting_arr, s=100, alpha=0.6, edgecolors='k', linewidth=1)
    for i, label in enumerate(labels):
        ax1.annotate(label, (sharpness_measure[i], forgetting_arr[i]),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    ax1.set_xlabel(r'$\frac{1}{2} \lambda_{max} \|\Delta w\|^2$', fontsize=14)
    ax1.set_ylabel('Forgetting (F₁)', fontsize=12)
    ax1.set_title('(c) Forgetting vs Sharpness (LoRA Subspace)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add trend line if enough points
    if len(sharpness_measure) > 2:
        z = np.polyfit(sharpness_measure, forgetting_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sharpness_measure.min(), sharpness_measure.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()

    # Plot (d): Eigenvalue spectrum comparison
    ax2 = axes[1]
    # This will be populated when we have eigenvalue data from multiple tasks
    ax2.set_xlabel('Eigenvalue Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=12)
    ax2.set_title('(d) Hessian Eigenvalues in LoRA Subspace', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.5, 'Eigenvalue spectra will be\nplotted when available',
            ha='center', va='center', transform=ax2.transAxes, fontsize=10, alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nForgetting analysis plot saved to {plot_path}")


def plot_eigenvalue_comparison(
    eigenvalue_dict: Dict,
    output_dir: str,
    filename: str = "eigenvalue_comparison.png"
):
    """
    Plot eigenvalue spectra for different tasks (Figure 2d style).

    Args:
        eigenvalue_dict: Dictionary with format:
                        {task_name: eigenvalues_array}
        output_dir: Output directory
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(eigenvalue_dict)))

    for (task_name, eigenvals), color in zip(eigenvalue_dict.items(), colors):
        indices = np.arange(1, len(eigenvals) + 1)
        ax.plot(indices, eigenvals, 'o-', label=task_name, color=color,
               linewidth=2, markersize=6)

    ax.set_xlabel('Eigenvalue Index', fontsize=12)
    ax.set_ylabel('Eigenvalue Magnitude', fontsize=12)
    ax.set_title('Hessian Eigenvalue Spectrum in LoRA Subspace', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Eigenvalue comparison plot saved to {plot_path}")


def analyze_loss_landscape_lora(
    model,
    dataloader,
    device,
    output_dir: str,
    name: str,
    initial_lora_params: torch.Tensor = None,
    num_eigenvalues: int = 20,
    compute_hessian: bool = True
) -> Dict:
    """
    Loss landscape analysis in LoRA subspace.
    This is specifically designed for LoRA-based continual learning.

    Following the paper's methodology (Eq. 5): F₁ ≈ (1/2) λ_max ||Δw||²

    Args:
        model: PEFT model with LoRA
        dataloader: DataLoader for computing loss
        device: Device
        output_dir: Output directory
        name: Name for saving files
        initial_lora_params: Initial LoRA parameters (for computing displacement)
        num_eigenvalues: Number of top eigenvalues to compute
        compute_hessian: Whether to compute Hessian eigenvalues

    Returns:
        Dictionary with analysis results
    """
    print(f"\nLoRA subspace loss landscape analysis for {name}...")

    # Get current LoRA parameters
    current_lora_params = get_lora_parameters(model, device)
    lora_dim = current_lora_params.numel()

    print(f"  LoRA parameter dimension: {lora_dim}")

    # Compute LoRA parameter displacement if initial params provided
    lora_param_norm = torch.norm(current_lora_params).item()
    lora_displacement = None

    if initial_lora_params is not None:
        lora_displacement = torch.norm(initial_lora_params - current_lora_params).item()
        print(f"  LoRA parameter displacement ||Δw_LoRA||: {lora_displacement:.4f}")
        print(f"  LoRA parameter norm ||w_LoRA||: {lora_param_norm:.4f}")

    # Compute top eigenvalues of Hessian in LoRA subspace
    eigenvalues = np.zeros(num_eigenvalues)
    lambda_max = 0.0

    if compute_hessian:
        print(f"  Computing top {num_eigenvalues} Hessian eigenvalues in LoRA subspace...")
        try:
            torch.cuda.empty_cache()
            eigenvalues = compute_hessian_eigenvalues_lora_subspace(
                model, dataloader, device, num_eigenvalues
            )
            lambda_max = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            print(f"  λ_max (LoRA): {lambda_max:.6f}")

            # Compute forgetting bound
            if lora_displacement is not None:
                forgetting_bound = 0.5 * lambda_max * (lora_displacement ** 2)
                print(f"  Forgetting bound (F₁ ≈ ½λ_max||Δw||²): {forgetting_bound:.6f}")

            # Save eigenvalues plot
            if len(eigenvalues) > 0 and np.any(eigenvalues > 0):
                plot_eigenvalues(eigenvalues, f"{name}_lora", output_dir)

        except Exception as e:
            print(f"  Error computing Hessian: {e}")
            eigenvalues = np.zeros(num_eigenvalues)
            lambda_max = 0.0
    else:
        print(f"  Skipping Hessian computation (compute_hessian=False)")

    metrics = {
        "lambda_max_lora": lambda_max,
        "eigenvalues_lora": eigenvalues.tolist(),
        "lora_param_norm": lora_param_norm,
        "lora_displacement": lora_displacement,
        "lora_dimension": lora_dim,
        "forgetting_bound": 0.5 * lambda_max * (lora_displacement ** 2) if lora_displacement else None
    }

    return {
        "current_lora_params": current_lora_params,
        "metrics": metrics
    }


def analyze_loss_landscape_efficient(
    model,
    dataloader,
    device,
    output_dir: str,
    name: str,
    initial_params: torch.Tensor = None,
    num_eigenvalues: int = 20,
    skip_hessian: bool = True  # Skip Hessian due to memory issues
) -> Dict:
    """
    Efficient loss landscape analysis using only eigenvalues and parameter norms.
    Based on Mirzadeh et al. 2020 Figure 2 and Figure 3.

    Args:
        model: Model to analyze
        dataloader: DataLoader for computing loss
        device: Device
        output_dir: Output directory
        name: Name for saving files
        initial_params: Initial parameters (for computing displacement)
        num_eigenvalues: Number of top eigenvalues to compute
        skip_hessian: Skip Hessian computation (saves memory)

    Returns:
        Dictionary with analysis results
    """
    print(f"\nEfficient loss landscape analysis for {name}...")

    # Get current parameters
    current_params = get_model_parameters(model, device)

    # Compute parameter displacement if initial params provided
    param_norm = torch.norm(current_params).item()
    displacement = None
    if initial_params is not None:
        displacement = compute_parameter_distance(initial_params, current_params)
        print(f"  Parameter displacement ||Δw||: {displacement:.2f}")
        print(f"  Parameter norm ||w||: {param_norm:.2f}")

    # Compute top eigenvalues of Hessian (optional, memory-intensive)
    eigenvalues = np.zeros(num_eigenvalues)
    lambda_max = 0.0

    if not skip_hessian:
        print(f"  Computing top {num_eigenvalues} Hessian eigenvalues...")
        try:
            # Clear cache before Hessian computation
            torch.cuda.empty_cache()
            eigenvalues = compute_hessian_eigenvalues(model, dataloader, device, num_eigenvalues)
            lambda_max = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            print(f"  λ_max: {lambda_max:.4f}")

            # Save eigenvalues plot
            if len(eigenvalues) > 0 and np.any(eigenvalues > 0):
                plot_eigenvalues(eigenvalues, name, output_dir)
        except Exception as e:
            print(f"  Skipping Hessian computation due to error: {e}")
            eigenvalues = np.zeros(num_eigenvalues)
            lambda_max = 0.0
    else:
        print(f"  Skipping Hessian computation (skip_hessian=True)")
        print(f"  Note: Focus on parameter displacement ||Δw|| for forgetting analysis")

    metrics = {
        "lambda_max": lambda_max,
        "eigenvalues": eigenvalues.tolist(),
        "param_norm": param_norm,
        "displacement": displacement
    }

    return {
        "current_params": current_params,
        "metrics": metrics
    }


def plot_eigenvalues(eigenvalues: np.ndarray, name: str, output_dir: str):
    """
    Plot eigenvalue spectrum.

    Args:
        eigenvalues: Array of eigenvalues
        name: Name for plot
        output_dir: Output directory
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Top {len(eigenvalues)} Hessian Eigenvalues - {name}')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, f"eigenvalues_{name}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Eigenvalue plot saved to {plot_path}")


def analyze_loss_landscape(
    model,
    dataloader,
    device,
    output_dir: str,
    name: str,
    distance: float = 1.0,
    num_points: int = 21,
    seed: int = 42
) -> Dict:
    """
    Complete loss landscape analysis for a model checkpoint.

    DEPRECATED: Use analyze_loss_landscape_efficient() for faster analysis.

    Args:
        model: Model to analyze
        dataloader: DataLoader for computing loss
        device: Device
        output_dir: Output directory
        name: Name for saving files
        distance: Maximum distance to explore
        num_points: Grid resolution
        seed: Random seed

    Returns:
        Dictionary with analysis results
    """
    # Get current parameters
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.data.view(-1).to(device))
    center_params = torch.cat(params)

    # Generate random directions
    dir1 = generate_random_direction(model, seed=seed, device=device)
    dir2 = generate_random_direction(model, seed=seed + 1, device=device)

    # Ensure directions are orthogonal
    dir2 = dir2 - (torch.dot(dir1, dir2) / torch.dot(dir1, dir1)) * dir1
    dir2 = dir2 / torch.norm(dir2)

    # Compute loss landscape
    alphas, betas, loss_grid = compute_2d_loss_landscape(
        model, dataloader, device, center_params,
        dir1, dir2, distance, num_points
    )

    # Plot
    plot_path = os.path.join(output_dir, f"landscape_{name}.png")
    plot_loss_landscape(alphas, betas, loss_grid, title=f"Loss Landscape - {name}", save_path=plot_path)

    # Compute metrics
    metrics = compute_sharpness_metrics(loss_grid, alphas, betas)

    return {
        "alphas": alphas,
        "betas": betas,
        "loss_grid": loss_grid,
        "metrics": metrics,
        "plot_path": plot_path
    }
