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
