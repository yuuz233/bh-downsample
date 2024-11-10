import pandas as pd
import numpy as np
from downsample import select_representative_samples_adaptive, evaluate_coverage
import matplotlib.pyplot as plt

def plot_comparison(original_data, selected_data, save_path=None):
    """
    Create a scatter plot comparing original and selected data distributions
    """
    plt.figure(figsize=(15, 5))
    
    # Original data plot
    plt.subplot(121)
    plt.scatter(original_data[:, 0], original_data[:, 1], 
                alpha=0.1, s=1, label='Original')
    plt.title(f'Original Data\n({len(original_data)} points)')
    plt.xlabel('X1 (normalized Hz)')
    plt.ylabel('X2 (normalized Watts)')
    
    # Selected data plot
    plt.subplot(122)
    plt.scatter(selected_data[:, 0], selected_data[:, 1],
                alpha=0.5, s=5, label='Selected')
    plt.title(f'Selected Data\n({len(selected_data)} points)')
    plt.xlabel('X1 (normalized Hz)')
    plt.ylabel('X2 (normalized Watts)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # File paths
    input_file = "data/data_set_1.csv"
    output_file = "output/ds1_selected_data.csv"
    plot_file = "output/ds1_comparison_plot.png"
    
    print("Reading input data...")
    # Read CSV without header, and with specific column names
    df = pd.read_csv(input_file, header=None, names=['index', 'X1', 'X2'])
    
    # Extract features
    X1 = df['X1'].values
    X2 = df['X2'].values
    
    print(f"Original data shape: {df.shape}")
    
    # Select representative samples (modified to get both samples and indices)
    print("Selecting representative samples...")
    selected_uniform, uniform_indices = select_representative_samples_adaptive(X1, X2, balance_weight=0.0, return_indices=True)
    selected_balanced, balanced_indices = select_representative_samples_adaptive(X1, X2, balance_weight=0.5, return_indices=True)
    selected_density, density_indices = select_representative_samples_adaptive(X1, X2, balance_weight=1.0, return_indices=True)
    
    # Move this line before the auto-tuning section
    original_data = np.column_stack((X1, X2))
    
    # Auto-tune custom balance weight using binary search
    # This is the back propagation part
    print("\nAuto-tuning custom balance weight...")
    left, right = 0.1, 0.9
    best_custom_weight = None
    best_custom_coverage = -1
    best_custom_samples = None
    best_custom_indices = None
    
    while right - left > 0.01:
        mid = (left + right) / 2
        left_weight = (2 * left + right) / 3
        right_weight = (left + 2 * right) / 3
        
        print(f"\nTrying balance weights: {left_weight:.3f} and {right_weight:.3f}")
        
        # Test left point
        selected_left, left_indices = select_representative_samples_adaptive(X1, X2, balance_weight=left_weight, return_indices=True)
        coverage_left = evaluate_coverage(original_data, selected_left)
        
        # Test right point
        selected_right, right_indices = select_representative_samples_adaptive(X1, X2, balance_weight=right_weight, return_indices=True)
        coverage_right = evaluate_coverage(original_data, selected_right)
        
        print(f"Coverage scores - Left: {coverage_left:.3f}, Right: {coverage_right:.3f}")
        
        # Update best result if needed
        if coverage_left > best_custom_coverage:
            best_custom_coverage = coverage_left
            best_custom_weight = left_weight
            best_custom_samples = selected_left
            best_custom_indices = left_indices
            
        if coverage_right > best_custom_coverage:
            best_custom_coverage = coverage_right
            best_custom_weight = right_weight
            best_custom_samples = selected_right
            best_custom_indices = right_indices
        
        # Update search interval
        if coverage_left > coverage_right:
            right = right_weight
        else:
            left = left_weight
    
    print(f"\nBest auto-tuned balance weight: {best_custom_weight:.3f} (coverage: {best_custom_coverage:.3f})")
    
    # Update custom results with auto-tuned values
    selected_custom, custom_indices = best_custom_samples, best_custom_indices
    coverage_custom = best_custom_coverage
    
    # Evaluate coverage
    coverage_uniform = evaluate_coverage(original_data, selected_uniform)
    coverage_balanced = evaluate_coverage(original_data, selected_balanced)
    coverage_density = evaluate_coverage(original_data, selected_density)
    
    print(f"\nCoverage score (uniform): {coverage_uniform:.2f}")
    print(f"Coverage score (balanced): {coverage_balanced:.2f}")
    print(f"Coverage score (density): {coverage_density:.2f}")
    print(f"Coverage score (custom/auto-tuned): {coverage_custom:.2f}")
    
    # Find the best performing method based on coverage score
    coverage_scores = {
        'balanced': (coverage_balanced, selected_balanced, balanced_indices),
        'density': (coverage_density, selected_density, density_indices),
        'custom': (coverage_custom, selected_custom, custom_indices)
    }
    
    best_method = max(coverage_scores.items(), key=lambda x: x[1][0])
    best_method_name = best_method[0]
    best_coverage = best_method[1][0]
    best_samples = best_method[1][1]
    best_indices = best_method[1][2]
    
    print(f"\nBest performing method: {best_method_name} (coverage: {best_coverage:.2f})")
    
    # Create visualization with best performing method
    print("Creating comparison plot...")
    plot_comparison(original_data, best_samples, plot_file)
    
    # Save both values and indices for best performing method
    print(f"Saving selected data using {best_method_name} method...")
    
    # Save the actual values
    selected_df = pd.DataFrame(best_samples, columns=['X1', 'X2'])
    selected_df.index = np.arange(1, len(selected_df) + 1)
    selected_df.to_csv(output_file, header=False, float_format='%.16f')
    
    # Save the indices
    indices_df = pd.DataFrame({'selected_indices': [i + 1 for i in best_indices]})  # Add 1 to match 1-based indexing
    indices_df.to_csv('output/ds1_selected_indices.csv', header=False, index=False)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 