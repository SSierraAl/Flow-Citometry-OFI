

#Borrar puntos usando el TSNE y el espectograma!!!!! nice nice nice

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import product

# Define parameter grid
param_grid = {
    'frame_length': [256, 512, 1024],
    'frame_step': [128, 256, 384],
    'fft_length': [256, 512, 1024]
}

# Create a list of all parameter combinations
param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

def extract_spectrogram_features(folder_path, frame_length, frame_step, fft_length, target_length=4095):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            signal = np.load(os.path.join(folder_path, filename))
            # Compute spectrogram with dynamic parameters
            spectrogram = tf.signal.stft(signal, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
            Sxx = tf.abs(spectrogram)
            Sxx_flat = Sxx.numpy().flatten()
            # Pad or truncate to target length
            if Sxx_flat.size < target_length:
                Sxx_flat = np.pad(Sxx_flat, (0, target_length - Sxx_flat.size))
            else:
                Sxx_flat = Sxx_flat[:target_length]
            data.append(Sxx_flat)
    return np.array(data)
# Collect data and labels for each particle size for each parameter combination
all_data = []
all_labels = []

for params in param_combinations:
    # Collect data for each particle size and parameter combination
    data_2um = [extract_spectrogram_features('D:/particles/WAV_2um_to_numpy', **params) for params in param_combinations]
    data_4um = [extract_spectrogram_features('D:/particles/WAV_4um_to_numpy', **params) for params in param_combinations]
    data_10um = [extract_spectrogram_features('D:/particles/WAV_10um_to_numpy', **params) for params in param_combinations]

    
    # Combine and label data for the current parameter set
    combined_data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
    combined_labels = np.array([0] * data_2um.shape[0] + [1] * data_4um.shape[0] + [2] * data_10um.shape[0])
    
    all_data.append(combined_data)
    all_labels.append(combined_labels)

# Loop through each parameter combination and visualize with t-SNE or PCA
for idx, params in enumerate(param_combinations):
    print(f"Testing parameters: {params}")

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data[idx])

    # Apply dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(scaled_data)
    labels = all_labels[idx]  # Use only labels for the current parameter set

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[labels == 0, 0], reduced_data[labels == 0, 1], label="2um", alpha=0.5)
    plt.scatter(reduced_data[labels == 1, 0], reduced_data[labels == 1, 1], label="4um", alpha=0.5)
    plt.scatter(reduced_data[labels == 2, 0], reduced_data[labels == 2, 1], label="10um", alpha=0.5)
    plt.title(f"t-SNE Visualization (frame_length={params['frame_length']}, frame_step={params['frame_step']}, fft_length={params['fft_length']})")
    plt.legend()
    plt.show()