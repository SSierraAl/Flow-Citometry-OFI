import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
import os

# Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to visualize spectrogram with filtered frequency range
def plot_filtered_spectrogram(file_path, sample_rate=2000000, lowcut=6000, highcut=60000, display_min_freq=8000, display_max_freq=35000):
    # Load the signal
    signal = np.load(file_path)
    
    # Apply bandpass filter
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, sample_rate)

    # Generate the spectrogram
    f, t, Sxx = spectrogram(filtered_signal, fs=sample_rate, window='blackman', nperseg=512, noverlap=384)

    # Filter the frequencies for display range
    f_idx = (f >= display_min_freq) & (f <= display_max_freq)
    f_display = f[f_idx]
    Sxx_display = Sxx[f_idx, :]

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f_display, 10 * np.log10(Sxx_display), shading='gouraud')
    plt.colorbar(label='Intensity (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram of {os.path.basename(file_path)} (Filtered 10kHz - 40kHz)')
    plt.ylim([display_min_freq, display_max_freq])
    plt.show()

# Folder path and file list
folder_path = 'D:/particles/10um_DB'  # Update to your folder path
file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])

# Loop through files, displaying each one after pressing Enter
for file_path in file_list:
    plot_filtered_spectrogram(file_path)
    input("Press Enter to see the next file...")
