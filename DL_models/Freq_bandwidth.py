import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd


################################################################################
def FFT_calc(datos, samplefreq):
    """
    Calculate the FFT (Fast Fourier Transform) of the input data.
    
    Parameters:
    datos : array-like
        Time-domain signal data.
    samplefreq : int
        Sampling frequency of the data in Hz.

    Returns:
    amplitude : array-like
        Amplitude spectrum of the signal.
    freq_fft : array-like
        Corresponding frequency values.
    phase : array-like
        Phase spectrum of the signal.
    """
    n = len(datos)  # Length of the data
    # Perform FFT and calculate amplitude and phase
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)  # Compute FFT
    freq_fft = np.fft.rfftfreq(len(datos), 1 / samplefreq)  # Corresponding frequencies
    amplitude = np.abs(fft_result)  # Magnitude of the FFT
    phase = np.angle(fft_result)  # Phase of the FFT

    return amplitude, freq_fft, phase




# Bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, order, fs):
    nyquist = 0.5 * fs
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = butter(order, [lowcut, highcut], btype='band')
    y = filtfilt(b, a, data)
    return y

# FFT calculation to find the peak frequency
def find_peak_frequency(signal, samplefreq):
    # Perform FFT

    amplitude, freq_fft,_ =FFT_calc(signal, samplefreq)
    # Find the peak frequency
    peak_idx = np.argmax(amplitude)
    peak_frequency = freq_fft[peak_idx]
    return peak_frequency

# Process folder to get peak frequencies
def process_folder(folder_path, samplefreq, lowcut, highcut, order):
    peak_frequencies = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        signal = np.load(file_path)
        
        # Apply bandpass filter
        filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, order, samplefreq)
        
        #Plot the max amplitude
        #peak_frequency=max(np.abs(filtered_signal))

        # Find the peak frequency
        peak_frequency = find_peak_frequency(filtered_signal, samplefreq)
        peak_frequencies.append(peak_frequency)
    
    return peak_frequencies

# Define parameters
sample_freq = 2000000  # Sampling frequency in Hz
lowcut = 8000          # Lower cutoff frequency for bandpass
highcut = 60000        # Upper cutoff frequency for bandpass
order = 4              # Filter order

# Paths to the folders (update these paths as needed)
folder_2um = 'D:/particles/signal_2um'
folder_4um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/HF_5_10_4um_doublet_particle'
folder_10um = 'D:/particles/signal_10um'

folder_2um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/HF_5_10_2um_doublet_Good'
folder_4um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/HF_5_10_4um_doublet_Good'
folder_10um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/HF_5_10_10um_doublet_Good'


folder_2um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/DB_2um'
folder_4um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/DB_4um'
folder_10um = 'C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/DB_10um'

folder_2um = 'D:/particles/2um_DB_Full' #2um_DB_full tambien funciona
folder_4um = 'D:/particles/DB_4um' #4_um_DB lento
folder_10um = 'D:/particles/10um_DB' #DB_10um funciona


# Process each folder to get the peak frequencies
peak_freq_2um = process_folder(folder_2um, sample_freq, lowcut, highcut, order)
peak_freq_4um = process_folder(folder_4um, sample_freq, lowcut, highcut, order)
peak_freq_10um = process_folder(folder_10um, sample_freq, lowcut, highcut, order)

# Calculate mean and standard deviation for each folder
mean_2um = np.mean(peak_freq_2um)
std_2um = np.std(peak_freq_2um)

mean_4um = np.mean(peak_freq_4um)
std_4um = np.std(peak_freq_4um)

mean_10um = np.mean(peak_freq_10um)
std_10um = np.std(peak_freq_10um)

# Prepare data for plotting
data = {
    'Folder': ['2um', '4um', '10um'],
    'Mean Frequency': [mean_2um, mean_4um, mean_10um],
    'Lower Bound': [mean_2um - std_2um, mean_4um - std_4um, mean_10um - std_10um],
    'Upper Bound': [mean_2um + std_2um, mean_4um + std_4um, mean_10um + std_10um]
}

df = pd.DataFrame(data)

# Plotting the candlestick chart
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each "candle" for each folder
for i in range(len(df)):
    ax.plot([i, i], [df['Lower Bound'][i], df['Upper Bound'][i]], color='black', lw=2)  # Deviation bar
    ax.plot(i, df['Mean Frequency'][i], 'bo', markersize=8)  # Mean point

# Customize plot
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Folder'])
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Mean and Standard Deviation of Peak Frequencies by Folder')
plt.grid()
plt.show()