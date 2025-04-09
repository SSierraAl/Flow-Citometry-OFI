import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.io import wavfile
from scipy.ndimage import shift
from scipy.signal import resample
from scipy.signal import decimate
from matplotlib.ticker import FuncFormatter
# Load signal function
def load_signal(file_path):
    return np.load(file_path)

# Data augmentation techniques
def add_noise(signal, noise_level=0.02):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise


def time_shift(signal, shift_factor=0.2):
    shift_samples = int(len(signal) * shift_factor)
    return np.roll(signal, shift_samples)

def invert_signal(signal):
    return -signal

def random_interpolation(signal, factor=0.1):
    num_points = int(len(signal) * factor)
    random_indices = np.random.choice(len(signal), num_points, replace=False)
    interpolated_signal = signal.copy()
    interpolated_signal[random_indices] = np.interp(random_indices, np.arange(len(signal)), signal)
    return interpolated_signal

def bitwise_downsample(signal, resolution=50):
    # Downsample each sample using the resolution R
    return np.floor(signal * resolution) / resolution

def sampling_rate_downsample(signal, k=3):
    # Overwrite each group of k samples with the first sample in the group
    #for i in range(0, len(signal), k):
    #    signal[i:i + k] = signal[i]


    signal_downsampled = decimate(signal, 4)
    return signal_downsampled

# Example folder path (modify for your case)
folder_path = 'D:/particles/Paper_DATA_2um'
file_name = random.choice([f for f in os.listdir(folder_path) if f.endswith('.npy')])
file_path = os.path.join(folder_path, file_name)

# Load original signal
original_signal = load_signal(file_path)

# Apply data augmentations
noisy_signal = add_noise(original_signal)
shifted_signal = time_shift(original_signal)
inverted_signal = invert_signal(original_signal)
interpolated_signal = random_interpolation(original_signal)
bitwise_signal = bitwise_downsample(original_signal, resolution=50)
sampling_downsampled_signal = sampling_rate_downsample(original_signal.copy(), k=3)



plt.plot(original_signal, color='royalblue')  # Use a blue color similar to 'Blues' cmap
  # Title with Times New Roman font
plt.xlabel('[mS]', fontsize=38, fontname='Times New Roman')  # Label X-axis with Times New Roman font
plt.ylabel('[mV]', fontsize=38, fontname='Times New Roman')  # Label Y-axis with Times New Roman font

# Increase font size for ticks
# Create a formatter to divide x-ticks by 1000
def divide_by_1000(x, pos):
    return x / 2000

# Set the x-axis labels formatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(divide_by_1000))
plt.xticks(fontsize=38, fontname='Times New Roman')
plt.yticks(fontsize=38, fontname='Times New Roman')

plt.plot(original_signal)
plt.grid()
plt.show()



# Plot all signals
plt.figure(figsize=(15, 12))

plt.subplot(7, 1, 1)
plt.plot(original_signal)
plt.title('Original Signal')

plt.subplot(7, 1, 2)
plt.plot(noisy_signal)
plt.title('Signal with Noise')

plt.subplot(7, 1, 3)
plt.plot(shifted_signal)
plt.title('Signal with Time Shift')

plt.subplot(7, 1, 4)
plt.plot(inverted_signal)
plt.title('Inverted Signal')

plt.subplot(7, 1, 5)
plt.plot(interpolated_signal)
plt.title('Random Interpolated Signal')

plt.subplot(7, 1, 6)
plt.plot(bitwise_signal)
plt.title('Bitwise Downsampled Signal (R=50)')

plt.subplot(7, 1, 7)
plt.plot(sampling_downsampled_signal)
plt.title('Sampling Rate Downsampled Signal (k=3)')

plt.tight_layout()
plt.show()
