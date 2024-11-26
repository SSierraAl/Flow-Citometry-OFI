import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.signal import spectrogram, butter, filtfilt
from itertools import product
from scipy.signal import decimate
# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to extract spectrogram features with given parameters
def extract_spectrogram_features(folder_path, sample_rate, display_min_freq, display_max_freq, nperseg, noverlap):
    features = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        signal = np.load(file_path)
        
        # Apply bandpass filter (fixed range)
        filtered_signal = butter_bandpass_filter(signal, 7000, 40000, sample_rate)
        filtered_signal = decimate(filtered_signal, 4)
        sample_rate=500000
        # Generate the spectrogram with specified parameters
        f, t, Sxx = spectrogram(filtered_signal, fs=sample_rate, window='blackman', nperseg=nperseg, noverlap=noverlap)

        # Extract relevant frequency range
        f_idx = (f >= display_min_freq) & (f <= display_max_freq)
        Sxx_display = Sxx[f_idx, :].flatten()  # Flatten the spectrogram data for this range
        
        features.append(Sxx_display)
    
    return np.array(features)

# Function to evaluate clustering accuracy with given parameters
def evaluate_clustering_accuracy(display_min_freq, display_max_freq, nperseg, noverlap, sample_rate=2000000):
    # Load data from each folder and extract features
    data_2um = extract_spectrogram_features('D:/particles/Paper_DATA_2um', sample_rate, display_min_freq, display_max_freq, nperseg, noverlap)
    data_4um = extract_spectrogram_features('D:/particles/Paper_DATA_4um', sample_rate, display_min_freq, display_max_freq, nperseg, noverlap)
    data_10um = extract_spectrogram_features('D:/particles/Paper_DATA_10um', sample_rate, display_min_freq, display_max_freq, nperseg, noverlap)

    # Combine data and create labels
    X = np.vstack((data_2um, data_4um, data_10um))
    y_true = np.array([0] * len(data_2um) + [1] * len(data_4um) + [2] * len(data_10um))

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=5)  # Using a consistent number of components for each test
    X_pca = pca.fit_transform(X_scaled)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++')
    kmeans.fit(X_pca)
    y_pred = kmeans.labels_

    # Map clusters to original labels
    def map_clusters(y_true, y_pred):
        from scipy.stats import mode
        labels = np.zeros_like(y_pred)
        for i in range(3):  # Assuming 3 clusters
            mask = (y_pred == i)
            labels[mask] = mode(y_true[mask])[0]
        return labels

    y_mapped = map_clusters(y_true, y_pred)

    # Calculate and return accuracy
    accuracy = accuracy_score(y_true, y_mapped)
    return accuracy

# Parameter ranges for grid search within the useful range
display_min_freq_values = [3000, 5000]
display_max_freq_values = [40000]
nperseg_values = [128, 256, 512, 500, 50]

# Grid search over parameter combinations with valid noverlap values
best_accuracy = 0
best_params = {}

for nperseg in nperseg_values:
    noverlap_values = [int(nperseg * 0.25), int(nperseg * 0.5), int(nperseg * 0.75)]  # Set valid noverlap values
    
    for display_min_freq, display_max_freq, noverlap in product(display_min_freq_values, display_max_freq_values, noverlap_values):
        if noverlap >= nperseg:
            continue  # Ensure noverlap is less than nperseg
        
        accuracy = evaluate_clustering_accuracy(display_min_freq, display_max_freq, nperseg, noverlap)
        print(f'Params: display_min_freq={display_min_freq}, display_max_freq={display_max_freq}, nperseg={nperseg}, noverlap={noverlap} -> Accuracy: {accuracy:.4f}')
        
        # Update best parameters if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'display_min_freq': display_min_freq,
                'display_max_freq': display_max_freq,
                'nperseg': nperseg,
                'noverlap': noverlap
            }

print("\nBest Accuracy:", best_accuracy)
print("Best Parameters:", best_params)

#Params: display_min_freq=7000, display_max_freq=40000, nperseg=512, noverlap=384 -> Accuracy: 0.6307