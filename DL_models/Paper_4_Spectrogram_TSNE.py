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


# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y


def load_files_from_folder(folder_path):
    """Load all numpy files from a given folder."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            signal = np.load(file_path)
            signal = butter_bandpass_filter(signal, 7000, 40000, 2000000)
            # Downsample the signal
            #signal = decimate(signal, 3)  # Downsample by a factor of 4
            data.append(signal)
    return data

def extract_spectrogram_features(folder_path, fs):
    """Extract spectrogram features for each signal in the folder."""
    signals = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=500, noverlap=375)


        display_max_freq = 40000
        display_min_freq= 5000
        # Extract relevant frequency range
        f_idx = (f >= display_min_freq) & (f <= display_max_freq)
        Sxx_display = Sxx[f_idx, :].flatten()  # Flatten the spectrogram data for this range
        # Flatten spectrogram to create a feature vector
        features.append(Sxx.flatten())


    return np.array(features)

# New sampling frequency after downsampling
fs_downsampled = 500000

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
