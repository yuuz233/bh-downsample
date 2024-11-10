import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm
import time

def select_representative_samples_fast(X1, X2, n_samples=2500, density_weight=0.5):
    """
    Faster version using MiniBatchKMeans and approximate density estimation
    """
    start_time = time.time()
    total_points = len(X1)
    print(f"\nProcessing {total_points:,} points to select {n_samples:,} samples...")
    
    # Combine features into array
    X = np.column_stack((X1, X2))
    
    # Number of samples for each strategy
    n_uniform = int(n_samples * (1 - density_weight))
    n_density = n_samples - n_uniform
    
    # Strategy 1: Uniform coverage using MiniBatchKMeans (much faster than regular KMeans)
    print(f"\nPhase 1: MiniBatch K-means clustering to select {n_uniform:,} uniform samples...")
    kmeans = MiniBatchKMeans(n_clusters=n_uniform, 
                            random_state=42, 
                            batch_size=1000,
                            verbose=0)
    
    with tqdm(total=100, desc="K-means progress") as pbar:
        kmeans.fit(X)
        pbar.update(100)
    
    uniform_samples = kmeans.cluster_centers_
    
    # Strategy 2: Approximate density-based selection using nearest neighbors
    print(f"\nPhase 2: Approximate density estimation to select {n_density:,} samples...")
    
    # Use a smaller subset for density estimation if dataset is very large
    sample_size = min(50000, total_points)
    if total_points > sample_size:
        print(f"Using {sample_size:,} random samples for density estimation...")
        indices = np.random.choice(total_points, sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Use number of neighbors as density estimate
    k = 50  # number of neighbors to consider
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    
    with tqdm(total=100, desc="Building neighbor tree") as pbar:
        nn.fit(X_sample)
        pbar.update(100)
    
    print("Calculating approximate densities...")
    distances, _ = nn.kneighbors(X)
    density_scores = -np.mean(distances, axis=1)  # negative mean distance as density
    
    # Select additional points based on density
    print("\nSelecting high-density points...")
    density_indices = np.argsort(density_scores)[-n_density:]
    density_samples = X[density_indices]
    
    # Combine both sets of samples
    selected_samples = np.vstack((uniform_samples, density_samples))
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing completed:")
    print(f"- Total time: {total_time:.2f} seconds")
    print(f"- Processing speed: {total_points/total_time:.0f} points/second")
    print(f"- Selected {len(selected_samples):,} points from {total_points:,} original points")
    
    return selected_samples

def select_representative_samples_fastest(X1, X2, n_samples=2500):
    """
    Extremely fast version using stratified sampling based on grid density
    """
    start_time = time.time()
    total_points = len(X1)
    print(f"\nProcessing {total_points:,} points to select {n_samples:,} samples...")
    
    # Combine features into array
    X = np.column_stack((X1, X2))
    
    # Create grid
    grid_size = 50
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    # Assign points to grid cells
    print("\nAssigning points to grid cells...")
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    x_indices = np.digitize(X[:, 0], x_bins) - 1
    y_indices = np.digitize(X[:, 1], y_bins) - 1
    
    # Count points in each cell
    grid_counts = np.zeros((grid_size, grid_size))
    cell_points = {}
    
    print("Calculating grid densities...")
    for i in tqdm(range(len(X))):
        x_idx, y_idx = x_indices[i], y_indices[i]
        grid_counts[x_idx, y_idx] += 1
        cell_key = (x_idx, y_idx)
        if cell_key not in cell_points:
            cell_points[cell_key] = []
        cell_points[cell_key].append(i)
    
    # Calculate number of samples per cell based on density
    total_count = np.sum(grid_counts)
    samples_per_cell = np.ceil((grid_counts / total_count) * n_samples).astype(int)
    
    # Select points from each cell
    print("\nSelecting representative points...")
    selected_indices = []
    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            if grid_counts[i, j] > 0:
                cell_key = (i, j)
                n_select = min(samples_per_cell[i, j], len(cell_points[cell_key]))
                selected_indices.extend(
                    np.random.choice(cell_points[cell_key], 
                                   size=n_select, 
                                   replace=False)
                )
    
    # Ensure exactly n_samples points are selected
    if len(selected_indices) > n_samples:
        selected_indices = np.random.choice(selected_indices, 
                                          size=n_samples, 
                                          replace=False)
    
    selected_samples = X[selected_indices]
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing completed:")
    print(f"- Total time: {total_time:.2f} seconds")
    print(f"- Processing speed: {total_points/total_time:.0f} points/second")
    print(f"- Selected {len(selected_samples):,} points from {total_points:,} original points")
    
    return selected_samples

def evaluate_coverage(original_data, selected_samples, grid_size=50):
    """
    Evaluate the coverage of selected samples compared to original data
    """
    print("\nEvaluating coverage...")
    # Create grid
    x_min, x_max = original_data[:, 0].min(), original_data[:, 0].max()
    y_min, y_max = original_data[:, 1].min(), original_data[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    
    # Count points in each grid cell
    def count_points_in_grid(data):
        grid_counts = np.zeros((grid_size, grid_size))
        for x, y in tqdm(data, desc="Calculating grid coverage"):
            i = int((x - x_min) / (x_max - x_min) * (grid_size - 1))
            j = int((y - y_min) / (y_max - y_min) * (grid_size - 1))
            grid_counts[i, j] += 1
        return grid_counts
    
    original_coverage = count_points_in_grid(original_data)
    selected_coverage = count_points_in_grid(selected_samples)
    
    coverage_score = np.sum(selected_coverage > 0) / np.sum(original_coverage > 0)
    return coverage_score

def select_representative_samples_adaptive(X1, X2, n_samples=2500, balance_weight=0.5, grid_size=50, return_indices=False):
    """
    Adaptive sampling that allows smooth transition between uniform and density-based selection
    
    Parameters:
    -----------
    X1, X2 : array-like
        Input features (normalized)
    n_samples : int, default=2500
        Number of samples to select
    balance_weight : float, default=0.5
        Weight between uniform (0.0) and density-based (1.0) sampling
        - 0.0: Purely uniform distribution
        - 0.5: Balanced between uniform and density-based
        - 1.0: Purely density-based distribution
    grid_size : int, default=50
        Size of the grid for density estimation
    """
    start_time = time.time()
    total_points = len(X1)
    print(f"\nProcessing {total_points:,} points to select {n_samples:,} samples...")
    
    # Combine features into array
    X = np.column_stack((X1, X2))
    
    # Create grid for density estimation
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    x_bins = np.linspace(x_min, x_max, grid_size + 1)  # Add +1 to match grid size
    y_bins = np.linspace(y_min, y_max, grid_size + 1)  # Add +1 to match grid size
    
    # Calculate grid densities
    print("\nCalculating grid densities...")
    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[x_bins, y_bins])
    
    # Normalize density
    density_probs = H / np.sum(H)
    
    # Create uniform probability
    uniform_probs = np.ones_like(density_probs) / np.sum(density_probs > 0)
    uniform_probs[density_probs == 0] = 0
    
    # Combine probabilities based on weight
    combined_probs = (1 - balance_weight) * uniform_probs + balance_weight * density_probs
    combined_probs = combined_probs / np.sum(combined_probs)
    
    # Assign points to grid cells
    print("\nAssigning points to grid cells...")
    x_indices = np.digitize(X[:, 0], x_bins) - 1
    y_indices = np.digitize(X[:, 1], y_bins) - 1
    
    # Create dictionary of points in each cell
    cell_points = {}
    for i in tqdm(range(len(X))):
        x_idx, y_idx = x_indices[i], y_indices[i]
        cell_key = (x_idx, y_idx)
        if cell_key not in cell_points:
            cell_points[cell_key] = []
        cell_points[cell_key].append(i)
    
    # Calculate samples per cell
    samples_per_cell = np.round(combined_probs * n_samples).astype(int)
    
    # Select points
    print("\nSelecting representative points...")
    selected_indices = []
    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            if samples_per_cell[i, j] > 0:
                cell_key = (i, j)
                if cell_key in cell_points:
                    n_select = min(samples_per_cell[i, j], len(cell_points[cell_key]))
                    if n_select > 0:
                        selected_indices.extend(
                            np.random.choice(cell_points[cell_key], 
                                           size=n_select, 
                                           replace=False)
                        )
    
    # Ensure exactly n_samples points are selected
    if len(selected_indices) > n_samples:
        selected_indices = np.random.choice(selected_indices, 
                                          size=n_samples, 
                                          replace=False)
    elif len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        available_indices = list(set(range(len(X))) - set(selected_indices))
        additional_indices = np.random.choice(available_indices, 
                                            size=remaining, 
                                            replace=False)
        selected_indices.extend(additional_indices)
    
    selected_samples = X[selected_indices]
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing completed:")
    print(f"- Total time: {total_time:.2f} seconds")
    print(f"- Processing speed: {total_points/total_time:.0f} points/second")
    print(f"- Selected exactly {len(selected_samples):,} points")
    print(f"- Balance weight: {balance_weight:.2f} (0=uniform, 1=density-based)")
    
    if return_indices:
        return selected_samples, selected_indices
    return selected_samples
