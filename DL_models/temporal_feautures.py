import numpy as np
import os
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from scipy.optimize import curve_fit
from scipy.signal import hilbert
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import os
from scipy.signal import find_peaks, hilbert

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def load_files_from_folder(folder_path):
    """Load all numpy files from a given folder."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            signal = np.load(file_path)
            data.append(signal)
    return data


def count_peaks_with_gaussian(signal, data_X, ampFFT1, min_amplitude=0.05):
    # Step 1: Preprocess the signal by setting all values less than zero to zero
    processed_signal = np.maximum(signal, 0)
    
    # Step 2: Perform Gaussian fitting on the envelope of the signal
    valid_P_y2 = np.abs(hilbert(signal))  # Hilbert transform for envelope
    try:
        # Fit Gaussian to the signal envelope
        optimized_params, _ = curve_fit(gaussian, data_X, valid_P_y2, maxfev=10000)
        amplitude, mean, stddev = optimized_params
        amplitude_normalized = (amplitude / np.max(amplitude)) * max(signal)
        y_curve = gaussian(data_X, amplitude_normalized, mean, stddev)
        # Define the threshold as max of y_curve divided by 10
        threshold = np.max(y_curve) / 10
        min_amplitude=np.max(y_curve)/5
    except Exception as e:
        print("Gaussian fit failed:", e)
        return 0, 0  # Return zero peaks if the fit fails

    gaussian_above_threshold = y_curve >= threshold
    width_points = np.sum(gaussian_above_threshold) / ampFFT1

    # Step 3: Count peaks based on Gaussian threshold and original criteria
    peak_count = 0
    i = 0
    n = len(processed_signal)
    
    while i < n:
        if processed_signal[i] == 0:
            i += 1
            continue

        # Start of a non-zero segment
        start = i
        while i < n and processed_signal[i] > 0:
            i += 1
        end = i

        # Check for peak amplitude and Gaussian threshold
        segment_max = np.max(processed_signal[start:end])
        if segment_max >= min_amplitude and np.any(y_curve[start:end] >= threshold):
            peak_count += 1

    return peak_count, width_points




def extract_temporal_features(signal, threshold):
    """Extract temporal features from a signal."""
    # Amplitude statistics

    signal = butter_bandpass_filter(signal, 7000, 80000, 4, 2000000)
    data_X = np.arange(len(signal))


    max_amplitude = np.max(signal)
    std_amplitude = np.std(signal)
 
    ampFFT1, freqfft1, _ = FFT_calc(signal, 2000000)


    num_bursts, passage = count_peaks_with_gaussian(signal, data_X, np.argmax(ampFFT1))
        



    # Pack all features into a dictionary
    features = {
        'max_amplitude': max_amplitude,
        'std_amplitude': std_amplitude,
        'num_bursts': num_bursts,
        'main_Freq': freqfft1[np.argmax(ampFFT1)],
        'Passage':passage*max_amplitude,

    }
    return features

# Load and process all signals in a folder
def extract_features_from_folder(folder_path, threshold=0.05):
    signals = load_files_from_folder(folder_path)
    features = []
    for signal in signals:
        features.append(extract_temporal_features(signal, threshold=threshold))
    return features

# Extract features for each particle type
#data_2um = extract_features_from_folder('D:/particles/2um_DB_Full')
#data_4um = extract_features_from_folder('D:/particles/DB_4um')
#data_10um = extract_features_from_folder('D:/particles/10um_DB')


# Extract features for each particle type
data_2um = extract_features_from_folder('D:/particles/WAV_2um_to_numpy')
data_4um = extract_features_from_folder('D:/particles/WAV_4um_to_numpy - copia')
data_10um = extract_features_from_folder('D:/particles/WAV_10um_to_numpy')


# Extract features for each particle type
#data_2um = extract_features_from_folder('D:/particles/DB_2um')
#data_4um = extract_features_from_folder('D:/particles/DB_4um')
#data_10um = extract_features_from_folder('D:/particles/DB_10um')



# Convert to a structured array (or DataFrame if using pandas)
import pandas as pd
data_2um_df = pd.DataFrame(data_2um)
data_2um_df['label'] = 0  # Label for 2um particles

data_4um_df = pd.DataFrame(data_4um)
data_4um_df['label'] = 1  # Label for 4um particles

data_10um_df = pd.DataFrame(data_10um)
data_10um_df['label'] = 2  # Label for 10um particles

# Combine all data
data_df = pd.concat([data_2um_df, data_4um_df, data_10um_df], ignore_index=True)
print(data_df.head())

# Find the minimum class size (i.e., the smallest class)
min_class_size = data_df['label'].value_counts().min()
# Downsample each class to the minimum class size
data_df = data_df.groupby('label').apply(lambda x: x.sample(n=min_class_size, random_state=42)).reset_index(drop=True)

# Verify the class distribution
print("Balanced Class Distribution:")
print(data_df['label'].value_counts())


# Split data into features and labels
X = data_df.drop('label', axis=1)
y = data_df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Check class distribution
print("Class Distribution in Dataset:")
print(data_df['label'].value_counts())

# Cross-validate to assess stability
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Calculate the mean of each feature for each category
feature_means = data_df.groupby('label').mean()

# Plot the mean features for each category
plt.figure(figsize=(12, 8))

# Loop through each feature and create a subplot for it
for i, feature in enumerate(feature_means.columns, 1):
    plt.subplot(2, 4, i)
    plt.bar(feature_means.index, feature_means[feature], tick_label=['2um', '4um', '10um'])
    plt.title(f'Mean {feature} by Category')
    plt.xlabel('Category')
    plt.ylabel(f'Mean {feature}')

plt.tight_layout()
plt.show()



# Scatter plot of max_amplitude vs std_amplitude, colored by category
plt.figure(figsize=(10, 8))

# Plot each category with a different color and label
for label, color in zip([0, 1, 2], ['blue', 'green', 'red']):
    subset = data_df[data_df['label'] == label]
    plt.scatter(subset['max_amplitude'], subset['Passage'], label=f'Category {label}', alpha=0.7, edgecolors='k')

# Add labels and title
plt.xlabel('Max Amplitude')
plt.ylabel('Passage')
plt.title('Scatter Plot of Max Amplitude vs. Std Amplitude by Category')
plt.legend()
plt.grid(True)
plt.show()