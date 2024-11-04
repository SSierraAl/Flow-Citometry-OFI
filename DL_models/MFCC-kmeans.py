import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import librosa
from itertools import product

# Function to extract MFCC features with given parameters
def extract_mfcc_features(folder_path, sample_rate, n_mfcc, n_fft, hop_length, fmin, fmax):
    features = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        signal = np.load(file_path)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
        
        # Flatten the MFCC array to create a feature vector
        mfccs_flattened = mfccs.flatten()
        features.append(mfccs_flattened)
    
    return np.array(features)

# Function to evaluate clustering accuracy with given parameters
def evaluate_clustering_accuracy(n_mfcc, n_fft, hop_length, fmin, fmax, sample_rate=2000000):
    # Load data from each folder and extract features
    data_2um = extract_mfcc_features('D:/particles/2um_DB', sample_rate, n_mfcc, n_fft, hop_length, fmin, fmax)
    data_4um = extract_mfcc_features('D:/particles/4um_DB', sample_rate, n_mfcc, n_fft, hop_length, fmin, fmax)
    data_10um = extract_mfcc_features('D:/particles/10um_DB', sample_rate, n_mfcc, n_fft, hop_length, fmin, fmax)

    # Combine data and create labels
    X = np.vstack((data_2um, data_4um, data_10um))
    y_true = np.array([0] * len(data_2um) + [1] * len(data_4um) + [2] * len(data_10um))

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=20)  # Using a consistent number of components for each test
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
# Parameter ranges for grid search
n_mfcc_values = [13, 20, 40]
n_fft_values = [512, 1024, 2048]
hop_length_values = [128, 256, 512]
fmin_values = [8000, 10000, 12000]
fmax_values = [30000, 35000, 40000]

# Grid search over all parameter combinations
best_accuracy = 0
best_params = {}

for n_mfcc, n_fft, hop_length, fmin, fmax in product(n_mfcc_values, n_fft_values, hop_length_values, fmin_values, fmax_values):
    accuracy = evaluate_clustering_accuracy(n_mfcc, n_fft, hop_length, fmin, fmax)
    print(f'Params: n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}, fmin={fmin}, fmax={fmax} -> Accuracy: {accuracy:.4f}')
    
    # Update best parameters if current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {
            'n_mfcc': n_mfcc,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'fmin': fmin,
            'fmax': fmax
        }

print("\nBest Accuracy:", best_accuracy)
print("Best Parameters:", best_params)
