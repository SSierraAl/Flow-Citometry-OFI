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


# Flag to decide whether to retrain the model
RETRAIN = True

# Parameters
columns = 65
channels = 1
EPOCHS = 400
LEARNING_RATE = 0.0006
BATCH_SIZE = 32
classes = 3 


# Function to load signals and compute spectrogram using TensorFlow Lite's `tf.signal.stft`
# Function to load signals and compute spectrogram using TensorFlow Lite's `tf.signal.stft`
def extract_spectrogram_features(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            signal = np.load(os.path.join(folder_path, filename))
            #signal = butter_bandpass_filter(signal, 8000, 40000, 4, 2000000)
            #signal = signal/max(signal)
            # Compute the spectrogram
            #signal = decimate(signal, 1)
            # Compute the spectrogram

            spectrogram = tf.signal.stft(signal, frame_length=512, frame_step=128, fft_length=512)
            Sxx = tf.abs(spectrogram)
            # Expand dimensions to add the channel dimension for compatibility
            Sxx = tf.expand_dims(Sxx, -1)  # Shape now is (rows, columns, channels)
            # Resize spectrogram to (63, columns) for CNN compatibility
            Sxx = tf.image.resize(Sxx, [63, columns])  # Resize to (63, columns, 1)
            data.append(Sxx.numpy())  # Convert to NumPy array if needed

    return np.array(data)


# Load and preprocess data
data_2um = extract_spectrogram_features('D:/particles/Paper_DATA_2um_Band_Filter')
data_4um = extract_spectrogram_features('D:/particles/Paper_DATA_4um_Band_Filter')
data_10um = extract_spectrogram_features('D:/particles/Paper_DATA_10um_Band_Filter')

labels_2um = np.zeros(data_2um.shape[0])  # Label 0 for 2um particles
labels_4um = np.ones(data_4um.shape[0])   # Label 1 for 4um particles
labels_10um = np.full(data_10um.shape[0], 2)  # Label 2 for 10um particles

# Concatenate data and labels
data = np.concatenate([data_2um, data_4um, data_10um], axis=0)
labels = np.concatenate([labels_2um, labels_4um, labels_10um], axis=0)

# Convert labels to one-hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


# Define the model
def create_model_90():
    model = Sequential()
    rows = X_train.shape[1]
    
    # Input Reshape
    model.add(Reshape((rows, columns, channels), input_shape=(rows, columns, channels)))

    # Convolutional Blocks
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.1))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.1))
    model.add(Dense(classes, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model



def create_model_simplified():
    model = Sequential()
    rows = X_train.shape[1]

    # Input Reshape
    model.add(Reshape((rows, columns, channels), input_shape=(rows, columns, channels)))

    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.4))

    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))  # Increase dropout here
    model.add(Dense(classes, activation='softmax'))

    opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


if RETRAIN:
    model = create_model_90()

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        #class_weight=class_weights,
        callbacks=[early_stopping, checkpoint_callback],
        verbose=2
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(validation_dataset, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predictions and evaluation
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(validation_dataset)
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
    model = tf.keras.models.load_model('best_model.keras')
    print("Loaded pre-trained model.")