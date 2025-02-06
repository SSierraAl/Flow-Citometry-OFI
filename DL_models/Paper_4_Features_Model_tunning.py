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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import product






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

    return np.array(data)


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

##################################################

# Define hyperparameters for grid search
Dense1_layer_values = [128] #[8, 16, 32, 64, 128, 256, 512]
Dense2_layer_values = [256] #16, 32, 64, 128, 256, 512]
p_dropout_values = [0.1]
LEARNING_RATE = 0.0006
EPOCHS = 50
BATCH_SIZE = 32
Augmented_Data = 50

# Load data
raw_data_2um = load_raw_signals('D:/particles/Paper_DATA_2um_augmented')
raw_data_4um = load_raw_signals('D:/particles/Paper_DATA_4um_augmented')
raw_data_10um = load_raw_signals('D:/particles/Paper_DATA_10um_augmented')


# Load extracted features
#features_2um = extract_features(raw_data_2um)
#features_4um = extract_features(raw_data_4um)
#features_10um = extract_features(raw_data_10um)

# Label data
labels_2um = np.zeros(raw_data_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(raw_data_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(raw_data_10um.shape[0], 2)  # Label 2 for 10um

# Combine features and labels
X = np.concatenate([raw_data_2um, raw_data_4um, raw_data_10um], axis=0)
y = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)


X = extract_features(X)

# **Feature Scaling**: Ensuring all features are in the same range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split data into training, validation, and testing
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw, Augmented_Data)

# Combine original and augmented training data
X_train = np.concatenate([X_train_raw, X_train_aug], axis=0)
y_train = np.concatenate([y_train_raw, y_train_aug], axis=0)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))
y_val = one_hot_encoder.fit_transform(y_val.reshape(-1, 1))

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Function to build model
def build_dense_model(dense1, dense2, dropout_rate):
    model = Sequential([
        Dense(dense1, activation='relu', kernel_regularizer=l2(0.0001), input_shape=(X_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(dense2, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

# Grid search setup
best_accuracy = 0
best_params = None
best_model = None
results = []

for dense1, dense2, p_dropout in product(Dense1_layer_values, Dense2_layer_values, p_dropout_values):
    print(f"Training with Dense1={dense1}, Dense2={dense2}, Dropout={p_dropout}")

    # Build and train the model
    model = build_dense_model(dense1, dense2, p_dropout)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[early_stopping], verbose=0)

    # Evaluate on validation data
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Validation accuracy: {val_accuracy:.4f}")

    results.append((dense1, dense2, p_dropout, val_accuracy))

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = (dense1, dense2, p_dropout)
        best_model = model

# Final evaluation
print("\nBest Validation Accuracy:", best_accuracy)
print("Best Parameters:", best_params)

# Model predictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute classification metrics
final_accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Print detailed classification report
print("\nFinal Model Performance Metrics:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}\n")

print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=['2um', '4um', '10um']))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['2um', '4um', '10um'], yticklabels=['2um', '4um', '10um'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()






