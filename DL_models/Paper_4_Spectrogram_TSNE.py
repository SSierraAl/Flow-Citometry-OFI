import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.signal import decimate
from scipy.signal import spectrogram, butter, filtfilt
from scipy.signal import decimate
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import random

# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y


# Data augmentation techniques ######################################
def data_augmentation(data, labels, total_augmented_samples):
    """
    Augments the dataset to generate a specific number of new samples.
    """
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

    def apply_random_augmentations(signal):
        augmentations = [
            lambda x: add_noise(x, noise_level=random.uniform(0.01, 0.05)),
            lambda x: time_shift(x, shift_factor=random.uniform(0.1, 0.5)),
            invert_signal,
            lambda x: random_interpolation(x, factor=random.uniform(0.05, 0.2)),
            lambda x: bitwise_downsample(x, resolution=random.randint(10, 100)),
        ]
        num_augmentations = random.randint(1, len(augmentations))
        selected_augmentations = random.sample(augmentations, num_augmentations)
        for augmentation in selected_augmentations:
            signal = augmentation(signal)
        return signal

    # Initialize containers
    data_aug, labels_aug = [], []
    
    # Number of augmented samples per class
    num_classes = len(np.unique(labels))
    samples_per_class = total_augmented_samples // num_classes

    # Perform augmentation per class
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        for _ in range(samples_per_class):
            idx = random.choice(class_indices)  # Randomly choose a sample from the class
            augmented_signal = apply_random_augmentations(data[idx])
            data_aug.append(augmented_signal)
            labels_aug.append(class_label)

    return np.array(data_aug), np.array(labels_aug)



def load_files_from_folder(folder_path):
    """Load all numpy files from a given folder."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            signal = np.load(file_path)
            signal = butter_bandpass_filter(signal, 8000, 40000, 2000000)
            # Downsample the signal
            #signal = decimate(signal, 4)  # Downsample by a factor of 4
            data.append(signal)

    data_aug, y_train_aug = data_augmentation(data, [0], 1)
    data_combined = np.concatenate([data_aug, data], axis=0)
    return data_combined

def extract_spectrogram_features(folder_path, fs):
    """Extract spectrogram features for each signal in the folder."""
    signals = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=64)


        display_max_freq = 50000
        display_min_freq= 4000
        # Extract relevant frequency range
        f_idx = (f >= display_min_freq) & (f <= display_max_freq)
        Sxx_display = Sxx[f_idx, :].flatten()  # Flatten the spectrogram data for this range
        # Flatten spectrogram to create a feature vector
        features.append(Sxx.flatten())


    return np.array(features)

# New sampling frequency after downsampling
fs_downsampled = 2000000

# Load and extract features for each category with downsampled signals
data_2um = extract_spectrogram_features('D:/particles/Paper_DATA_2um_augmented', fs_downsampled)
data_4um = extract_spectrogram_features('D:/particles/Paper_DATA_4um_augmented', fs_downsampled)
data_10um = extract_spectrogram_features('D:/particles/Paper_DATA_10um_augmented', fs_downsampled)

# Label data
labels_2um = np.zeros(data_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(data_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(data_10um.shape[0], 2)  # Label 2 for 10um particles

# Concatenate all data and labels
data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

# Set font to Times New Roman globally
plt.rcParams["font.family"] = "Times New Roman"

# Define unique markers for each particle type
markers = {0: 'o', 1: '^', 2: 's'}  # Circles, triangles, squares
colors = ['blue', 'orange', 'green']  # Colors for each label

# Plotting t-SNE results with customizations
plt.figure(figsize=(12, 10))
particlesss=[2,4,10]
for label in np.unique(labels):
    label_data = data_tsne[labels == label]  # Filter data for this label
    plt.scatter(
        label_data[:, 0], label_data[:, 1],
        c=colors[int(label)],
        label=f"{particlesss[int(label)]}μm Particles",
        marker=markers[int(label)],
        alpha=0.7,
        edgecolor='k',  # Add black edges to markers
        s=100  # Adjust marker size
    )

# Increase axis number sizes
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Customize labels and title
plt.xlabel('t-SNE Component 1', fontsize=28)
plt.ylabel('t-SNE Component 2', fontsize=28)
#plt.title('t-SNE Visualization of Spectrogram Features', fontsize=18)

# Add legend
plt.legend(fontsize=26, loc='best')

# Show grid for better visual clarity
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()
