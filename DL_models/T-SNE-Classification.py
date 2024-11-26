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




def load_files_from_folder(folder_path):
    """Load all numpy files from a given folder."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            signal = np.load(file_path)
            data.append(signal)
    return data
#Params: display_min_freq=7000, display_max_freq=40000, nperseg=512, noverlap=384 -> Accuracy: 0.6307
def extract_spectrogram_features(folder_path, fs):
    """Extract spectrogram features for each signal in the folder."""
    signals = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        #signal=signal/max(signal)
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=512, noverlap=384)
        # Flatten spectrogram to create a feature vector
        features.append(Sxx.flatten())
    return np.array(features)

# Load and extract features for each category
data_2um = extract_spectrogram_features('D:/particles/Paper_DATA_2um')
data_4um = extract_spectrogram_features('D:/particles/Paper_DATA_4um')
data_10um = extract_spectrogram_features('D:/particles/Paper_DATA_10um')

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

# Plotting t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1, 2], label="Particle Type")
plt.title('t-SNE visualization of spectrogram features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()





# Separar datos de t-SNE por etiquetas
tsne_2um = data_tsne[labels == 0]
tsne_4um = data_tsne[labels == 1]
tsne_10um = data_tsne[labels == 2]

plt.figure(figsize=(10, 8))

# Graficar puntos de t-SNE con colores para cada grupo
plt.scatter(tsne_2um[:, 0], tsne_2um[:, 1], alpha=0.5, label="2um", s=30)
plt.scatter(tsne_4um[:, 0], tsne_4um[:, 1], alpha=0.5, label="4um", s=30)
plt.scatter(tsne_10um[:, 0], tsne_10um[:, 1], alpha=0.5, label="10um", s=30)


# Separar las componentes de t-SNE en función de las etiquetas
tsne_2um = data_tsne[labels == 0]
tsne_4um = data_tsne[labels == 1]
tsne_10um = data_tsne[labels == 2]

def coef_variation(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Verificación de condiciones para un CV confiable
    if mean == 0 or abs(mean) < 1e-5:  # Evitar divisiones con medias cercanas a cero
        return np.nan  # Retorna NaN para indicar un CV no fiable
    
    cv = (std_dev / mean) * 100
    
    # Validación adicional para valores extremos
    if abs(cv) > 1000:  # Umbral para evitar valores de CV inusuales
        return np.nan
    return cv

# Calcular CV para cada categoría y cada componente
cv_comp1_2um = coef_variation(tsne_2um[:, 0])
cv_comp1_4um = coef_variation(tsne_4um[:, 0])
cv_comp1_10um = coef_variation(tsne_10um[:, 0])

cv_comp2_2um = coef_variation(tsne_2um[:, 1])
cv_comp2_4um = coef_variation(tsne_4um[:, 1])
cv_comp2_10um = coef_variation(tsne_10um[:, 1])

# Plot histogram and CV for each PCA component
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Histogram and CV of the first t-SNE component
axs[0].hist(tsne_2um[:, 0], bins=20, alpha=0.5, label=f'2um (CV={cv_comp1_2um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[0].hist(tsne_4um[:, 0], bins=20, alpha=0.5, label=f'4um (CV={cv_comp1_4um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[0].hist(tsne_10um[:, 0], bins=20, alpha=0.5, label=f'10um (CV={cv_comp1_10um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[0].set_title('Histograma de la Componente t-SNE 1')
axs[0].set_xlabel('Componente 1')
axs[0].set_ylabel('Frecuencia')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Histogram and CV of the second t-SNE component
axs[1].hist(tsne_2um[:, 1], bins=20, alpha=0.5, label=f'2um (CV={cv_comp2_2um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[1].hist(tsne_4um[:, 1], bins=20, alpha=0.5, label=f'4um (CV={cv_comp2_4um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[1].hist(tsne_10um[:, 1], bins=20, alpha=0.5, label=f'10um (CV={cv_comp2_10um:.2f}%)', edgecolor='gray', linewidth=0.8)
axs[1].set_title('Histograma de la Componente t-SNE 2')
axs[1].set_xlabel('Componente 2')
axs[1].set_ylabel('Frecuencia')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()