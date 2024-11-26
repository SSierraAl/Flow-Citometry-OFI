import numpy as np
import os
import random
from scipy.signal import decimate

# Define augmentation techniques
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
    return np.floor(signal * resolution) / resolution

def sampling_rate_downsample(signal, k=3, original_length=None):
    downsampled_signal = decimate(signal, k)
    if original_length:  # Resize back to the original length
        return np.resize(downsampled_signal, original_length)
    return downsampled_signal

# Ensure all signals are the same length
def resize_signal(signal, target_length):
    if len(signal) != target_length:
        return np.resize(signal, target_length)
    return signal

# Apply random combination of augmentations
def apply_random_augmentations(signal):
    original_length = len(signal)  # Keep track of the original signal length
    augmentations = [
        lambda x: add_noise(x, noise_level=random.uniform(0.01, 0.05)),
        lambda x: time_shift(x, shift_factor=random.uniform(0.1, 0.5)),
        invert_signal,
        lambda x: random_interpolation(x, factor=random.uniform(0.05, 0.2)),
        lambda x: bitwise_downsample(x, resolution=random.randint(10, 100)),
        #lambda x: sampling_rate_downsample(x, k=random.randint(2, 4), original_length=original_length),
    ]
    num_augmentations = random.randint(1, len(augmentations))
    selected_augmentations = random.sample(augmentations, num_augmentations)
    for augmentation in selected_augmentations:
        signal = augmentation(signal)
    return resize_signal(signal, original_length)  # Ensure the output length is the same

# Augment dataset
def augment_dataset(input_folder, output_folder, num_new_files):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    for i in range(num_new_files):
        file_name = random.choice(original_files)
        file_path = os.path.join(input_folder, file_name)
        original_signal = np.load(file_path)

        augmented_signal = apply_random_augmentations(original_signal)

        output_file_path = os.path.join(output_folder, f"augmented_2um_{i}.npy")
        np.save(output_file_path, augmented_signal)

    print(f"{num_new_files} new augmented files saved to {output_folder}")

# Set paths and parameters
input_folder = 'D:/particles/Paper_DATA_2um'
output_folder = 'D:/particles/Paper_DATA_2um_augmented'
num_new_files = 200

# Run augmentation
#Uncomment!!!!
#augment_dataset(input_folder, output_folder, num_new_files)
