
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
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
import tensorflow as tf
import numpy as np
import time
import os
import tracemalloc
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
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
from tensorflow.keras.layers import Add



# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Data augmentation function
def data_augmentation(data, labels, total_augmented_samples):
    def apply_random_augmentations(signal):
        augmentations = [
            lambda x: x + np.random.normal(0, 0.02, x.shape),  # Add Gaussian noise
            lambda x: np.roll(x, int(len(x) * 0.2)),  # Shift signal
            lambda x: -x,  # Invert signal
            lambda x: np.interp(np.linspace(0, len(x), len(x)), np.arange(len(x)), x),  # Resample
            lambda x: np.floor(x * 50) / 50,  # Quantize
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

# Load raw signals
def load_raw_signals(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            signal = np.load(os.path.join(folder_path, filename))
            signal = decimate(signal, 10)
            signal = signal *10# / np.max(np.abs(signal))  # Normalize
            data.append(signal)
    return np.array(data)

# Load data
raw_data_2um = load_raw_signals('D:/particles/Paper_DATA_2um_augmented')
raw_data_4um = load_raw_signals('D:/particles/Paper_DATA_4um_augmented')
raw_data_10um = load_raw_signals('D:/particles/Paper_DATA_10um_augmented')

# Label data
labels_2um = np.zeros(raw_data_2um.shape[0])  
labels_4um = np.ones(raw_data_4um.shape[0])   
labels_10um = np.full(raw_data_10um.shape[0], 2)  

# Combine signals and labels
X = np.concatenate([raw_data_2um, raw_data_4um, raw_data_10um], axis=0)
y = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

# Split data
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw, 150)
X_train = np.concatenate([X_train_raw, X_train_aug], axis=0)
y_train = np.concatenate([y_train_raw, y_train_aug], axis=0)

# Ensure consistent input shape
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

# Function to build 1D CNN Model
def build_cnn_model(filter_size, kernel_size, dense_size, dropout_rate):
    model = Sequential([
        Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(filters=filter_size * 2, kernel_size=kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(filters=filter_size * 4, kernel_size=kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(dense_size, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0006), metrics=['accuracy'])
    return model

# Grid Search Setup
EPOCHS = 150
FILTER_SIZES = [64]
KERNEL_SIZES = [5]
DENSE_SIZES = [256]
DROPOUT_RATES = [0.2]

best_accuracy = 0
best_params = None
best_model = None

for filter_size, kernel_size, dense_size, dropout_rate in product(FILTER_SIZES, KERNEL_SIZES, DENSE_SIZES, DROPOUT_RATES):
    print(f"Training with Filters={filter_size}, Kernel={kernel_size}, Dense={dense_size}, Dropout={dropout_rate}")
    model = build_cnn_model(filter_size, kernel_size, dense_size, dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=[early_stopping], verbose=0)
    
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = (filter_size, kernel_size, dense_size, dropout_rate)
        best_model = model

# Final Evaluation
print("\nBest Validation Accuracy:", best_accuracy)
print("Best Parameters:", best_params)

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

final_accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print("\nFinal Model Performance Metrics:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}\n")

print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=['2um', '4um', '10um']))

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['2um', '4um', '10um'], yticklabels=['2um', '4um', '10um'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


def convert_to_tflite(model, quantization_type="float16"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply quantization
    if quantization_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Default quantization
    elif quantization_type == "float16":
        converter.target_spec.supported_types = [tf.float16]  # Float16 precision
    elif quantization_type == "hybrid":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    # Save the model
    tflite_model_path = f"quantized_model_{quantization_type}.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model converted to {quantization_type} TFLite format and saved as {tflite_model_path}")
    return tflite_model_path

def measure_tflite_latency(tflite_model_path, X_sample, num_runs=100):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    X_sample = X_sample.astype(np.float32)  # Ensure correct dtype

    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], X_sample)
        interpreter.invoke()
    avg_latency = (time.time() - start_time) / num_runs

    print(f"Average inference latency (TFLite): {avg_latency:.6f} seconds")
    return avg_latency


def measure_tflite_ram_usage(tflite_model_path, X_sample):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    
    tracemalloc.start()
    interpreter.set_tensor(input_details[0]['index'], X_sample)
    interpreter.invoke()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Peak RAM usage during TFLite inference: {peak / 1e6:.2f} MB")

def measure_tflite_size(tflite_model_path):
    size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # Convert to MB
    print(f"Quantized model storage size: {size:.2f} MB")
    return size

def run_tflite_inference(tflite_model_path, X_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    
    for i in range(len(X_test)):
        sample_input = X_test[i:i+1].astype(np.float32)  # Convert to float32 if necessary
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output_data))  # Get predicted class

    return np.array(y_pred)


def evaluate_tflite_model(tflite_model_path, X_test, y_test):
    y_pred_classes = run_tflite_inference(tflite_model_path, X_test)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    print("\n--- Quantized Model Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_classes, target_names=['2um', '4um', '10um']))


if __name__ == "__main__":
    # Convert the model to TFLite (Choose quantization type: 'float16', 'int8', or 'hybrid')
    quantization_type = "int8"  # Try 'int8' or 'hybrid' for different results
    tflite_model_path = convert_to_tflite(best_model, quantization_type)
    print("\n--- Evaluating Quantized Model Accuracy ---")
    evaluate_tflite_model(tflite_model_path, X_test, np.argmax(y_test, axis=1)) 
    # Generate a dummy input sample
    sample_input = np.random.rand(1, X_train.shape[1], 1).astype(np.float32)

    print("\n--- Evaluating Quantized Model ---")
    measure_tflite_size(tflite_model_path)
    measure_tflite_latency(tflite_model_path, sample_input)
    measure_tflite_ram_usage(tflite_model_path, sample_input)
