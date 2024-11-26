import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import mplcursors
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.signal import spectrogram, butter, filtfilt
from itertools import product

# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y
# Function to load numpy files and associate them with the array index
def load_files_from_folder(folder_path):
    files = []
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            files.append(file_path)
            signal = np.load(file_path)
            data.append(signal)
    return np.array(data), files
from scipy.signal import spectrogram, butter, filtfilt
# Extract spectrogram features and keep associated files
def extract_spectrogram_features(folder_path, fs=2000000):
    signals, files = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        signal = butter_bandpass_filter(signal, 7000, 40000, fs)
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=512, noverlap=384)
        features.append(Sxx.flatten())
    return np.array(features), files

# Load and label data
data_2um, files_2um = extract_spectrogram_features('D:/particles/Paper_DATA_2um_augmented')
data_4um, files_4um = extract_spectrogram_features('D:/particles/Paper_DATA_4um_augmented')
data_10um, files_10um = extract_spectrogram_features('D:/particles/Paper_DATA_10um_augmented')

# Labels
labels_2um = np.zeros(data_2um.shape[0])  
labels_4um = np.ones(data_4um.shape[0])   
labels_10um = np.full(data_10um.shape[0], 2)  

# Concatenate data, labels, and file paths
data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)
files = files_2um + files_4um + files_10um  # Complete list of file paths

# Standardize and apply t-SNE
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

# Plot t-SNE data
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1, 2], label="Particle Type")
plt.title('t-SNE visualization of spectrogram features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Use mplcursors to select points on the plot
cursor = mplcursors.cursor(scatter, hover=True)

# Variable to store the index of the last selected point
last_selected_index = None

@cursor.connect("add")
def on_add(sel):
    global last_selected_index
    # Get the index of the selected point
    target = sel.target
    index = np.where((data_tsne == target).all(axis=1))[0]
    if len(index) > 0:
        last_selected_index = index[0]  # Save the index of the selected point
        file_to_delete = files[last_selected_index]

        # Update annotation to show click-to-delete message
        sel.annotation.set_text(
            f"File: {os.path.basename(file_to_delete)}\nClick to delete this file"
        )

# Define a function to handle file deletion when clicking on the selected point
def on_click(event):
    global last_selected_index
    if last_selected_index is not None:  # Check if a point has been selected
        file_to_delete = files[last_selected_index]
        try:
            os.remove(file_to_delete)
            print(f"Deleted file: {file_to_delete}")
            # Hide the point visually by setting its size to 0
            scatter.set_sizes(np.where((data_tsne == data_tsne[last_selected_index]).all(axis=1), 0, scatter.get_sizes()))
            fig.canvas.draw()  # Refresh the plot
            last_selected_index = None  # Reset the selected index after deletion
        except Exception as e:
            print(f"Error deleting file {file_to_delete}: {e}")

# Connect the click event to the plot
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
