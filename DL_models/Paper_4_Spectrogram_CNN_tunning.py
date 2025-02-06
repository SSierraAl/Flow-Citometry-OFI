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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
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

from itertools import product


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.signal import decimate
from itertools import product
import random 

# Hyperparameters
RETRAIN = True
columns = 65
channels = 1
EPOCHS = 50  # Reduced for quick testing
classes = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0006
Augmented_Data=50
# Grid search parameters
nperseg_values = [128]  # 128, 256, 512
noverlap_factors = [0.5]  # Fraction of nperseg
Dense_layer_values = [8] # 8, 16, 32, 64, 128
Dense2_layer_values = [16] # 8, 16, 32, 64, 128
p_dropout_values = [0.1] #0.1, 0.2

# Function to load signals and compute spectrogram using TensorFlow Lite's `tf.signal.stft`
def extract_features_with_params(signals, nperseg, noverlap):
    if len(signals) == 0:
        print("Warning: Received empty input for feature extraction!")
        return np.array([])  # Return empty array to avoid further errors

    data = []
    for signal in signals:
        spectrogram = tf.signal.stft(signal, frame_length=nperseg, frame_step=noverlap, fft_length=nperseg)
        Sxx = tf.abs(spectrogram)

        if Sxx.shape[0] == 0:  # Ensure STFT result is not empty
            print("Warning: Empty STFT output!")
            continue

        num_bins = Sxx.shape[1]
        frequency_per_bin = 500000 / (2 * num_bins)
        min_bin = int(4000 / frequency_per_bin)
        max_bin = int(50000 / frequency_per_bin)

        # Slice spectrogram to desired frequency range
        if max_bin > num_bins:
            print("Warning: Adjusting max_bin index to match STFT size.")
            max_bin = num_bins

        Sxx = Sxx[:, min_bin:max_bin]  
        Sxx = tf.expand_dims(Sxx, -1)
        Sxx = tf.image.resize(Sxx, [63, columns])
        data.append(Sxx.numpy())

    return np.array(data) if data else np.array([]) 

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




# Load raw data
raw_data_2um = load_raw_signals('D:/particles/Paper_DATA_2um_augmented')
raw_data_4um = load_raw_signals('D:/particles/Paper_DATA_4um_augmented')
raw_data_10um = load_raw_signals('D:/particles/Paper_DATA_10um_augmented')

# Assign Labels
labels_2um = np.zeros(raw_data_2um.shape[0])  
labels_4um = np.ones(raw_data_4um.shape[0])   
labels_10um = np.full(raw_data_10um.shape[0], 2)  

# Combine data and labels
raw_data = np.concatenate([raw_data_2um, raw_data_4um, raw_data_10um], axis=0)
raw_labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

print('--- Data loaded ---')

# Split raw data into training and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(raw_data, raw_labels, test_size=0.3, random_state=42)

# Apply data augmentation only on training data
X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw, Augmented_Data)
print('--- Data augmented ---')

# Combine original and augmented training data
X_train_combined = np.concatenate([X_train_raw, X_train_aug], axis=0)
y_train_combined = np.concatenate([y_train_raw, y_train_aug], axis=0)

# Grid search
best_accuracy = 0
best_params = None
best_model = None
results = []

for nperseg, noverlap_factor, dense1, dense2, p_dropout in product(nperseg_values, noverlap_factors, Dense_layer_values, Dense2_layer_values, p_dropout_values):
    noverlap = int(nperseg * noverlap_factor)  

    print(f"Processing with nperseg={nperseg}, noverlap={noverlap}, Dense1={dense1}, Dense2={dense2}, p_dropout={p_dropout}")

    # Generate spectrograms
    X_train = extract_features_with_params(X_train_combined, nperseg, noverlap)
    X_test = extract_features_with_params(X_test_raw, nperseg, noverlap)

    # One-hot encode labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_train = one_hot_encoder.fit_transform(y_train_combined.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test_raw.reshape(-1, 1))

    # Split training data into training and validation sets
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_new, y_train_new)).shuffle(1000).batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Model function
    def build_dense_model():
        model = Sequential([
            Flatten(input_shape=(63, 65, 1)),
            Dense(dense1, activation='relu', kernel_regularizer=l2(0.0002)),
            BatchNormalization(),
            Dropout(p_dropout),
            Dense(dense2, activation='relu', kernel_regularizer=l2(0.0003)),
            BatchNormalization(),
            Dropout(p_dropout),
            Dense(classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
        return model

    # Train the model
    model = build_dense_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[early_stopping], verbose=0)

    # Evaluate and record results
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Validation accuracy for this configuration: {val_accuracy:.4f}")
    
    results.append((nperseg, noverlap, dense1, dense2, p_dropout, val_accuracy))

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = (nperseg, noverlap, dense1, dense2, p_dropout)
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















"""
    # Define the model with current parameters
    def create_model():
        model = Sequential()
        rows = X_train.shape[1]
        model.add(Reshape((rows, columns, channels), input_shape=(rows, columns, channels)))
        model.add(Conv2D(conv_1l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(p_dropout))
        model.add(Conv2D(conv_2l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(p_dropout))
        model.add(Flatten())
        model.add(Dense(conv_2l, activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(Dropout(p_dropout))
        model.add(Dense(classes, activation='softmax'))
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
"""