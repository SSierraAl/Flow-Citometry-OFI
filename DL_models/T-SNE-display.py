import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
def extract_spectrogram_features(folder_path, fs=2_000_000):
    """Extract spectrogram features for each signal in the folder."""
    signals = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=512, noverlap=384)
        # Flatten spectrogram to create a feature vector
        features.append(Sxx.flatten())
    return np.array(features)

# Load and extract features for each category
data_2um = extract_spectrogram_features('D:/particles/2um_DB_Full')
data_4um = extract_spectrogram_features('D:/particles/DB_4um')
data_10um = extract_spectrogram_features('D:/particles/10um_DB')

# Label data
labels_2um = np.zeros(data_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(data_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(data_10um.shape[0], 2)  # Label 2 for 10um particles

# Concatenate all data and labels
data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Crear el pipeline con PCA y RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normaliza los datos
    ('pca', PCA(n_components=50)),  # Reduce dimensionalidad con PCA
    ('classifier', RandomForestClassifier(random_state=42))  # Clasificador
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Hacer predicciones y evaluar el modelo
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Mostrar el reporte de clasificación
print("Classification Report:")
print(classification_report(y_test, y_pred))


"""
# Load and extract features from the new folder for 4um particles
data_4um_new = extract_spectrogram_features('D:/particles/4um_DB')

# Predict using the trained pipeline
y_pred_new = pipeline.predict(data_4um_new)

# Assuming these are all 4um particles, set true labels to 1
y_true_new = np.ones(data_4um_new.shape[0])

# Calculate accuracy for the new dataset
accuracy_new = accuracy_score(y_true_new, y_pred_new)
print(f"New Data Accuracy (4um): {accuracy_new:.4f}")

# Predict using the trained pipeline
y_pred_new = pipeline.predict(data_4um_new)

# Display the predicted categories for each sample in the new dataset
print("Predicted categories for each sample in NEW_TestDB_4um:")
print(y_pred_new)

#print(classification_report(y_true_new, y_pred_new))
"""
