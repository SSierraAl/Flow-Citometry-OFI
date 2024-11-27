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


# Hyperparameters
RETRAIN = True
columns = 65
channels = 1
EPOCHS = 50  # Reduced for quick testing
classes = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0006

# Grid search parameters
nperseg_values = [128, 256, 512]
noverlap_factors = [0.5]  # Fraction of nperseg
conv_1l_values = [8, 16, 32]
conv_2l_values = [16, 32, 64]
p_dropout_values = [0.1, 0.2]

# Function to load signals and compute spectrogram using TensorFlow Lite's `tf.signal.stft`
def extract_features_with_params(folder_path, nperseg, noverlap):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            signal = np.load(os.path.join(folder_path, filename))
            # Decimate the signal
            
            # New sampling frequency
            signal = butter_bandpass_filter(signal, 8000, 40000, 4, 2000000)
            signal = decimate(signal, 4)
            fs_decimated = 2000000 / 4
                        
            spectrogram = tf.signal.stft(signal, frame_length=nperseg, frame_step=noverlap, fft_length=nperseg)
            Sxx = tf.abs(spectrogram)
            Sxx = tf.expand_dims(Sxx, -1)
            Sxx = tf.image.resize(Sxx, [63, columns])
            data.append(Sxx.numpy())
    return np.array(data)

# Prepare labels
data_2um, data_4um, data_10um = None, None, None
labels_2um = None
labels_4um = None
labels_10um = None

# Grid search
best_accuracy = 0
best_params = None
results = []

for nperseg, conv_1l, conv_2l, p_dropout in product(nperseg_values, conv_1l_values, conv_2l_values, p_dropout_values):
    noverlap = int(nperseg * noverlap_factors[0])  # Calculate overlap
    
    # Update spectrogram features
    print(f"Processing with nperseg={nperseg}, noverlap={noverlap}, conv_1l={conv_1l}, conv_2l={conv_2l}, p_dropout={p_dropout}")
    data_2um = extract_features_with_params('D:/particles/Paper_DATA_2um_augmented', nperseg, noverlap)
    data_4um = extract_features_with_params('D:/particles/Paper_DATA_4um_augmented', nperseg, noverlap)
    data_10um = extract_features_with_params('D:/particles/Paper_DATA_10um_augmented', nperseg, noverlap)
    
    # Concatenate data and labels
    data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
    labels_2um = np.zeros(data_2um.shape[0])
    labels_4um = np.ones(data_4um.shape[0])
    labels_10um = np.full(data_10um.shape[0], 2)
    labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    
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
    
    # Train the model
    model = create_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate and record results
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Validation accuracy for this configuration: {val_accuracy:.4f}")
    results.append((nperseg, noverlap, conv_1l, conv_2l, p_dropout, val_accuracy))
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = (nperseg, noverlap, conv_1l, conv_2l, p_dropout)

# Print best parameters
print("Best Validation Accuracy:", best_accuracy)
print("Best Parameters:", best_params)

# Plot results (optional)
results = sorted(results, key=lambda x: x[5], reverse=True)
print("Top 5 configurations:")
for config in results[:5]:
    print(f"nperseg={config[0]}, noverlap={config[1]}, conv_1l={config[2]}, conv_2l={config[3]}, p_dropout={config[4]}: val_accuracy={config[5]:.4f}")