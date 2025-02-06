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

#############################################################################
# Model and Data Parameters
#############################################################################

# Flag to decide whether to retrain the model
RETRAIN = True
columns = 65
channels = 1
EPOCHS = 200
classes = 3 
BATCH_SIZE = 32
Augmented_Data=200

# Parameters to tunne in the gridsearch
LEARNING_RATE = 0.0006
conv_1l=8
conv_2l=16
p_dropout=0.1
#Spectrogram parameters
Nperseg= 128
Noverlap= 64


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


def plot_signal_and_spectrogram(signal, spectrogram, title="Signal and Spectrogram"):

    sample_rate = 500000  # Effective sampling rate after decimation
    spectrogram = spectrogram.numpy().T
    # Create the figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Plot raw signal
    ax[0].plot(signal, color='blue')
    ax[0].set_title(f"Raw Signal - {title}")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    # Plot spectrogram
    img = ax[1].imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    ax[1].set_title("Spectrogram")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax[1], label="Amplitude")

    plt.tight_layout()
    plt.show()


# Extract spectrogram features from raw signals ####################
def extract_spectrogram_features_from_raw(data):
    spectrograms = []
    for signal in data:
        #NEW
        # Compute the spectrogram
        spectrogram = tf.signal.stft(signal, frame_length=Nperseg, frame_step=Noverlap, fft_length=Nperseg)
        Sxx = tf.abs(spectrogram)
        # To plot ####################
        #Sxx = tf.abs(spectrogram).numpy()
        # Calculate frequency bins
        num_bins = Sxx.shape[1]
        frequency_per_bin = 500000 / (2 * num_bins)  # Nyquist limit
        min_bin = int(4000 / frequency_per_bin)
        max_bin = int(50000 / frequency_per_bin)
        # Slice spectrogram to desired frequency range
        Sxx = Sxx[:,min_bin:max_bin]  # Retain only desired rows
        #plot_signal_and_spectrogram(signal, Sxx, title="Signal and Spectrogram")
        #print(Sxx.shape)
        # Expand dimensions for compatibility and resize
        Sxx = tf.expand_dims(Sxx, -1)  # Add channel dimension (for CNN input)
        Sxx = tf.image.resize(Sxx, [63, columns])  # Resize for CNN compatibility
        spectrograms.append(Sxx.numpy())  # Convert to NumPy and store

    return np.array(spectrograms)


# Load data
raw_data_2um = load_raw_signals('D:/particles/Paper_DATA_2um_augmented')
raw_data_4um = load_raw_signals('D:/particles/Paper_DATA_4um_augmented')
raw_data_10um = load_raw_signals('D:/particles/Paper_DATA_10um_augmented')
# Labels for raw data
labels_2um = np.zeros(raw_data_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(raw_data_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(raw_data_10um.shape[0], 2)  # Label 2 for 10um particles
# Combine raw data and labels
raw_data = np.concatenate([raw_data_2um, raw_data_4um, raw_data_10um], axis=0)
raw_labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)
print('--- Data loaded ---')
# Split raw data into training and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(raw_data, raw_labels, test_size=0.3, random_state=42)

# Augment only the training data
X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw, Augmented_Data)
print('--- Data augmented ---')

# Combine original and augmented training data
X_train_combined = np.concatenate([X_train_raw, X_train_aug], axis=0)
y_train_combined = np.concatenate([y_train_raw, y_train_aug], axis=0)

# Generate spectrograms for training and testing datasets
X_train = extract_spectrogram_features_from_raw(X_train_combined)
X_test = extract_spectrogram_features_from_raw(X_test_raw)
print('--- Spectrogram features ---')

# Convert labels to one-hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train = one_hot_encoder.fit_transform(y_train_combined.reshape(-1, 1))
y_test = one_hot_encoder.transform(y_test_raw.reshape(-1, 1))

# Split training data into training and validation sets
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_new, y_train_new)).shuffle(1000).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Flatten the data for t-SNE
X_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten each sample
y_labels = np.argmax(y_train, axis=1)  # Convert one-hot labels to class indices

# Use a subset of data for t-SNE (e.g., 1000 samples)
subset_size = min(1000, X_flat.shape[0])
X_subset = X_flat[:subset_size]
y_subset = y_labels[:subset_size]

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_subset)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='viridis', s=15)
plt.colorbar(scatter, label='Class Label')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()
"""

####################################################################################################
# Models 
####################################################################################################
# 2D CNN 2 conv layers
def create_model_90():
    model = Sequential()
    rows = X_train.shape[1]
    
    # Input Reshape
    model.add(Reshape((rows, columns, channels), input_shape=(rows, columns, channels)))

    # Convolutional Blocks
    model.add(Conv2D(conv_1l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(p_dropout))

    model.add(Conv2D(conv_2l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(p_dropout))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(conv_2l, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(p_dropout))
    model.add(Dense(classes, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

# 2D CNN 1 conv layers
def create_model_1_2D_CNN():
    model = Sequential()
    rows = X_train.shape[1]
    model.add(Reshape((rows, columns, channels), input_shape=(rows, columns, channels)))

    # Single Convolutional Block
    model.add(Conv2D(conv_1l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(p_dropout))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(conv_1l, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(p_dropout))
    model.add(Dense(classes, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

#1D CNN 1conv layer
def create_model_1DCNN():
    
    model = Sequential()
    rows = X_train.shape[1]
    # Input Reshape for 1D CNN
    model.add(Reshape((rows, columns), input_shape=(rows, columns, channels)))

    # 1D Convolutional Block
    model.add(Conv1D(filters=conv_1l, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(p_dropout))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(conv_1l, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(p_dropout))
    model.add(Dense(classes, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_dense_model():
    model = Sequential()

    # Flatten input
    
    model.add(Flatten(input_shape=(63, 65, 1)))
    # First Dense Block
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(p_dropout))

    # Second Dense Block (optional)
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(p_dropout))

    # Output Layer
    model.add(Dense(classes, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

if RETRAIN:

    model = build_dense_model()
    # Callbacks
    modelname_params='best_model'+'.keras'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(modelname_params, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        verbose=2
    )
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(validation_dataset, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predictions and evaluation
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(test_dataset)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['2um', '4um', '10um'], yticklabels=['2um', '4um', '10um'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=['2um', '4um', '10um']))

else:
    # Load pre-trained model
    model = tf.keras.models.load_model('best_model_spectrogram.keras')
    print("Loaded pre-trained model.")