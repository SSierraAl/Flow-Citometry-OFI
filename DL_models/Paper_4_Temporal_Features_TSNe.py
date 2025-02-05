import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.signal import decimate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import SpatialDropout2D
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
from scipy.optimize import curve_fit

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


columns = 65
#Spectrogram parameters
Nperseg= 128
Noverlap= 64
# New sampling frequency after downsampling
fs_downsampled = 500000

def compute_power(signal):
    return np.mean(signal**2)



noisy=np.load('average_noise_2Mhz.npy')
noisy = decimate(noisy, 4)
avg_noisy_power = compute_power(noisy)


# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def data_augmentation(data, labels, total_augmented_samples):
    """
    Augments the dataset to generate a specific number of new samples.
    """
    def apply_random_augmentations(signal):
        augmentations = [
            lambda x: x + np.random.normal(0, 0.02, x.shape),
            lambda x: np.roll(x, int(len(x) * 0.2)),
            lambda x: -x,
            lambda x: np.interp(np.linspace(0, len(x), len(x)), np.arange(len(x)), x),
            lambda x: np.floor(x * 50) / 50,
        ]
        for _ in range(random.randint(1, len(augmentations))):
            signal = random.choice(augmentations)(signal)
        return signal

    data_aug, labels_aug = [], []
    num_classes = len(np.unique(labels))
    samples_per_class = total_augmented_samples // num_classes

    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        for _ in range(samples_per_class):
            idx = random.choice(class_indices)
            augmented_signal = apply_random_augmentations(data[idx])
            data_aug.append(augmented_signal)
            labels_aug.append(class_label)

    return np.array(data_aug), np.array(labels_aug)

# Load numpy files ##################################################
def load_raw_signals(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            signal = np.load(os.path.join(folder_path, filename))
            signal = butter_bandpass_filter(signal, 8000, 40000, 4, 2000000)
            signal = decimate(signal, 4)  # Decimate to reduce sampling rate 500k
            data.append(signal)

    data_aug, y_train_aug = data_augmentation(data, [0], 1)
    data_combined = np.concatenate([data_aug, data], axis=0)

    return np.array(data_combined)


def extract_spectrogram_features_from_raw(data):
    spectrograms = []
    for signal in data:
        spectrogram = tf.signal.stft(signal, frame_length=Nperseg, frame_step=Noverlap, fft_length=Nperseg)
        Sxx = tf.abs(spectrogram).numpy()

        num_bins = Sxx.shape[1]
        frequency_per_bin = fs_downsampled / (2 * num_bins)
        min_bin = int(4000 / frequency_per_bin)
        max_bin = int(50000 / frequency_per_bin)
        Sxx = Sxx[:, min_bin:max_bin]  

        Sxx = np.expand_dims(Sxx, -1)
        Sxx = tf.image.resize(Sxx, [63, columns]).numpy()
        spectrograms.append(Sxx)

    return np.array(spectrograms)

#Features to be extracted

def Get_Signal_Amplitude(data):
    # Smooth the signal by averaging 5 consecutive points
    smoothed_signal = np.convolve(data, np.ones(5)/5, mode='valid')
    # Get the maximum absolute amplitude
    max_amplitude = np.max(np.abs(smoothed_signal))
    return max_amplitude

def Get_Passage_Time(data):
    # Calculate the envelope of the signal (using Hilbert transform)
    analytic_signal = np.abs(hilbert(np.abs(data)))
    # Fit a Gaussian to the envelope
    x = np.arange(len(analytic_signal))
    try:
        params, _ = curve_fit(gaussian, x, analytic_signal, p0=[np.max(analytic_signal), np.argmax(analytic_signal), 10])
        # Calculate the width of the Gaussian where it's above 10% of the maximum
        threshold = 0.1 * params[0]
        width = 2 * params[2] * np.sqrt(-2 * np.log(threshold / params[0]))
    except Exception as e:
        print("Gaussian fit failed:", e)
        return 0 # Return zero peaks if the fit fails
    return width

def get_peak_spectral_power(data):
    # Compute the spectrogram
    spectrogram = tf.signal.stft(data, frame_length=Nperseg, frame_step=Noverlap, fft_length=Nperseg)
    Sxx = tf.abs(spectrogram)
    # Get the maximum spectral power
    peak_power = np.max(Sxx.numpy())
    spectral_energy= Sxx ** 2  # Square the magnitude to get power
    spectral_energy=  np.sum(spectral_energy) 
    return peak_power,spectral_energy

def compute_power(signal):
    return np.mean(signal**2)

def get_average_power(data):
    avg_power = compute_power(data)
    avg_power=avg_power-avg_noisy_power
    return avg_power


def get_get_doppler(data):
    amplitude, freq_fft, phase= FFT_calc(data, 500000)
    return freq_fft[np.argmax(amplitude)]


# Load data
raw_data_2um = load_raw_signals('D:/particles/Paper_DATA_2um_augmented')
raw_data_4um = load_raw_signals('D:/particles/Paper_DATA_4um_augmented')
raw_data_10um = load_raw_signals('D:/particles/Paper_DATA_10um_augmented')


# Extract features for each dataset
def extract_features(data):
    features = []
    for signal in data:
        amplitude = Get_Signal_Amplitude(signal)
        passage_time = Get_Passage_Time(signal)
        average_power= get_average_power(signal)       
        peak_power,spectral_energy = get_peak_spectral_power(signal)
        doppler_peak_freq= get_get_doppler(signal)

        features.append([amplitude, passage_time,average_power, peak_power,spectral_energy,doppler_peak_freq])

    return np.array(features)

features_2um = extract_features(raw_data_2um)
features_4um = extract_features(raw_data_4um)
features_10um = extract_features(raw_data_10um)

# Label data
labels_2um = np.zeros(features_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(features_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(features_10um.shape[0], 2)  # Label 2 for 10um \


# Concatenate all data and labels
data = np.concatenate([features_2um, features_4um, features_10um], axis=0)
labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)


# Plotting t-SNE results
plt.rcParams["font.family"] = "Times New Roman"
markers = {0: 'o', 1: '^', 2: 's'}  # Circles, triangles, squares
colors = ['blue', 'orange', 'green']  # Colors for each label
particle_sizes = [2, 4, 10]

plt.figure(figsize=(12, 10))
for label in np.unique(labels):
    label_data = data_tsne[labels == label]
    plt.scatter(
        label_data[:, 0], label_data[:, 1],
        c=colors[int(label)],
        label=f"{particle_sizes[int(label)]}μm Particles",
        marker=markers[int(label)],
        alpha=0.7,
        edgecolor='k',
        s=100
    )

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel('t-SNE Component 1', fontsize=28)
plt.ylabel('t-SNE Component 2', fontsize=28)
plt.legend(fontsize=26, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()








# Train a Random Forest Classifier to evaluate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(data_scaled, labels)

# Get feature importances from Random Forest
feature_importances = rf.feature_importances_

# Compute permutation importance
perm_importance = permutation_importance(rf, data_scaled, labels, scoring='accuracy', random_state=42)

# Feature names corresponding to extracted features
feature_names = ['Amplitude', 'Passage Time', 'Average Power', 'Peak Power', 'Spectral Energy', 'Doppler Peak Freq']

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]

# Plot feature importance from Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
plt.xlabel("Feature Importance (Random Forest)")
plt.ylabel("Features")
plt.title("Feature Importance for Particle Classification")
plt.show()

# Plot feature importance from Permutation Importance
sorted_perm_idx = np.argsort(perm_importance.importances_mean)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance.importances_mean[sorted_perm_idx], y=np.array(feature_names)[sorted_perm_idx], palette="magma")
plt.xlabel("Permutation Importance")
plt.ylabel("Features")
plt.title("Permutation Feature Importance for Particle Classification")
plt.show()