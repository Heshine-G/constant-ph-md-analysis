import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def read_xvg(filename):
    """Reads an XVG file, skipping header lines."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            try:
                data.append(list(map(float, line.strip().split())))
            except ValueError:
                continue
    return np.array(data)

def autocorrelation(x):
    """Computes the autocorrelation of a series using FFT."""
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0] # Normalize
    return acf

def process_file(filepath):
    """Reads and processes a single XVG file to get its autocorrelation."""
    try:
        data = read_xvg(filepath)
        if data.shape[1] < 2:
            raise ValueError("Data must have at least two columns.")
        time = data[:, 0] / 1000.0  # Convert time from ps to ns
        lambda_vals = data[:, 1]
        acf = autocorrelation(lambda_vals)
        # Clean up the label for the title
        label = os.path.basename(filepath).replace('.xvg', '').replace('_', ' ').title()
        return label, time[:len(acf)], acf, None
    except Exception as e:
        return os.path.basename(filepath), None, None, str(e)

def plot_as_subplots(results):
    """
    Plots each autocorrelation function in its own subplot within a grid.
    This is the clearest method for many lines.
    """
    num_plots = len(results)
    if num_plots == 0:
        print("No data to plot.")
        return
        
    # Determine the grid size (e.g., 5 columns)
    cols = 5
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), constrained_layout=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    for i, (label, time, acf) in enumerate(results):
        ax = axes[i]
        ax.plot(time, acf, color='teal')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (ns)", fontsize=8)
        ax.set_ylabel("ACF", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Autocorrelation of Lambda Coordinates", fontsize=16, weight='bold')
    plt.savefig("all_residues_subplots.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def main():
    xvg_files = sorted(glob.glob("*.xvg"))
    print(f"Found {len(xvg_files)} XVG files to process.")

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in xvg_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            label, time, acf, error = future.result()
            if error:
                print(f"Error processing file {futures[future]}: {error}")
                continue
            results.append((label, time, acf))
    
    # Sort results alphabetically by label for consistent plotting order
    results.sort(key=lambda x: x[0])
    
    # Plotting
    plot_as_subplots(results)
    
if __name__ == "__main__":
    main()