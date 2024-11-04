import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
#n_mfcc=13, n_fft=2048, hop_length=512, fmin=12000, fmax=35000 -> Accuracy: 0.7804
#Best Accuracy: 0.8640864086408641
#Best Parameters: {'n_mfcc': 13, 'n_fft': 512, 'hop_length': 128, 'fmin': 8000, 'fmax': 40000}
# Function to extract and plot MFCCs for a single file
def plot_mfcc(file_path, sample_rate=2000000, n_mfcc=13, n_fft=512, hop_length=128, fmin=8000, fmax=35000):
    # Load the signal
    signal = np.load(file_path)
    
    # Compute MFCCs with specified parameters
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    
    # Plot the MFCCs
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, hop_length=hop_length)
    plt.colorbar(label='MFCC Coefficients (dB)')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time (s)')
    
    plt.title(f'MFCC of {os.path.basename(file_path)}')
    plt.show()

# Folder path and file list
folder_path = 'D:/particles/10um_DB'  # Update to your folder path
file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])

# Loop through files, displaying each one after pressing Enter
for file_path in file_list:
    plot_mfcc(file_path)
    input("Press Enter to see the next file...")
