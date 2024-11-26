import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.signal import decimate
from scipy.signal import spectrogram, butter, filtfilt
from scipy.signal import decimate
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter

# FFT Calculation Function
def FFT_calc(data, fs):
    N = len(data)
    freq = np.fft.rfftfreq(N, d=1/fs)  # Positive frequencies
    fft_amplitude = np.abs(np.fft.rfft(data))  # FFT amplitudes
    return fft_amplitude, freq, N

# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y

# Load signals and calculate main frequency peaks
def process_folder(folder_path, fs, downsample_factor):
    data = []
    main_peaks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            signal = np.load(file_path)
            signal = butter_bandpass_filter(signal, 7000, 80000, fs)
            signal_downsampled = decimate(signal, downsample_factor)
            data.append(signal_downsampled)
            
            # Calculate FFT
            ampFFT, freqFFT, _ = FFT_calc(signal_downsampled, fs / downsample_factor)
            main_peak = freqFFT[np.argmax(ampFFT)]  # Frequency with max amplitude
            main_peaks.append(main_peak)
    return main_peaks

# Sampling and downsampling parameters
fs = 2000000  # Original sampling frequency
downsample_factor = 4
fs_downsampled = fs / downsample_factor

# Process each folder
peaks_2um = process_folder('D:/particles/Paper_DATA_2um', fs, downsample_factor)
peaks_4um = process_folder('D:/particles/Paper_DATA_4um', fs, downsample_factor)
peaks_10um = process_folder('D:/particles/Paper_DATA_10um', fs, downsample_factor)

# Plot distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(peaks_2um, label="2μm Particles", bw_adjust=0.5)
sns.kdeplot(peaks_4um, label="4μm Particles", bw_adjust=0.5)
sns.kdeplot(peaks_10um, label="10μm Particles", bw_adjust=0.5)
plt.title("Distribution of Main Frequency Peaks by Particle Type")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Density")
plt.legend()
plt.show()