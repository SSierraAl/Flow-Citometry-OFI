import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import os
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

# Function to compute average power
def compute_power(signal):
    return np.mean(signal**2)

# Function to compute SNR
def compute_snr(signal, noise):
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise)

# Step 1: Load and average noise files
def load_and_average_noise(noise_files):
    all_noise = []
    for file in noise_files:
        data = np.load(file)
        data=data[0:2500]
        data = butter_bandpass_filter(data, 4000, 50000, 4, 2000000)
        all_noise.append(data)
    average_noise = np.mean(all_noise, axis=0)
    np.save('average_noise_2Mhz.npy',average_noise, )
    return average_noise

# Step 2: Load and process signal files
def process_signals(signal_paths, x_noise):
    results = []
    for path in signal_paths:
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
        for file in files:
            x_noisy = np.load(file)

            # Check and adjust shapes if needed
            if x_noisy.shape != x_noise.shape:
                if x_noise.ndim == 1 and x_noisy.ndim == 2:
                    x_noise = x_noise[:, np.newaxis]  # Align dimensions
                elif x_noisy.shape[0] != x_noise.shape[0]:
                    raise ValueError(f"Shape mismatch: x_noisy {x_noisy.shape}, x_noise {x_noise.shape}")

            # Subtract noise
            x_clean = x_noisy - x_noise
            snr_clean = compute_snr(x_clean, x_noise)
            results.append((os.path.basename(path), snr_clean))
    return results

# Step 3: Generate candlestick plot

# Update font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

def plot_candlestick(snr_results, output_file):
    # Replace category names with specific labels
    labels = ['10µm', '4µm', '2µm']
    data = {}
    for category, snr in snr_results:
        if category not in data:
            data[category] = []
        data[category].append(snr)

    categories = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, category in enumerate(categories):
        snrs = np.array(data[category])
        mean_val = np.mean(snrs)
        min_val = np.min(snrs)
        max_val = np.max(snrs)
        std_val = np.std(snrs)

        # Plot min-max range (wick)
        ax.plot([i + 1, i + 1], [min_val, max_val], color='blue', lw=2)

        # Plot mean ± std (body)
        rect = patches.Rectangle((i + 0.8, mean_val - std_val), 0.4, 2 * std_val,
                                 linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect)

        # Plot mean value as a horizontal line
        ax.plot([i + 0.8, i + 1.2], [mean_val, mean_val], color='red', lw=2)

    # Set the x-axis with desired labels
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(labels,fontsize=20)  # Use the specific labels
    ax.tick_params(axis='y', labelsize=20) 
    ax.set_ylabel('SNR (dB)', fontsize=22)
    ax.grid(True)

    plt.tight_layout()

    # Save the plot as a high-quality vector graphic
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Noise file paths
    noise_files = ['D:/particles/WAV_noise_numpy/HFocusing_5_10_05um_0_158.npy',
                   'D:/particles/WAV_noise_numpy/HFocusing_5_10_05um_0_336.npy',
                   'D:/particles/WAV_noise_numpy/HFocusing_5_10_05um_0_662.npy',
                   'D:/particles/WAV_noise_numpy/HFocusing_5_10_05um_0_446.npy']
    
    # Signal paths
    signal_paths = ['D:/particles/Paper_DATA_10um_Band_Filter', 
                    'D:/particles/Paper_DATA_4um_Band_Filter', 
                    'D:/particles/Paper_DATA_2um_Band_Filter']

    # Compute average noise
    x_noise = load_and_average_noise(noise_files)

    # Process signal files
    snr_results = process_signals(signal_paths, x_noise)

    # Plot candlestick chart
    plot_candlestick(snr_results,output_file="candlestick_chart.pdf")




    