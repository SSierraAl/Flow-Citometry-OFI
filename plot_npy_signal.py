import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Function for designing the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function for applying the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# FFT calculation function
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
    print(n)
    # Perform FFT and calculate amplitude and phase
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)  # Compute FFT
    freq_fft = np.fft.rfftfreq(len(datos), 1 / samplefreq)  # Corresponding frequencies
    amplitude = np.abs(fft_result)  # Magnitude of the FFT
    phase = np.angle(fft_result)  # Phase of the FFT

    return amplitude, freq_fft, phase

# Load the .npy file
data = np.load('D:/particles/Paper_DATA_4um/HFocusing_5_10_4um_0_4.npy1838.npy')

print(len(data))
ASasAS

# Filter parameters
lowcut = 11000  # Lower cutoff frequency (Hz)
highcut = 80000  # Upper cutoff frequency (Hz)
fs = 2000000  # Sampling frequency in Hz (adjust to the sampling frequency of your data)

# Apply the bandpass filter
filtered_data = bandpass_filter(data, lowcut, highcut, fs, order=5)

# Compute the FFT of the filtered signal
amplitude, freq_fft, phase = FFT_calc(filtered_data, fs)

# Check if the array is 1D to plot
if data.ndim == 1:
    plt.figure(figsize=(10, 8))
    
    # Plot the original signal
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot the filtered signal
    plt.subplot(3, 1, 2)
    plt.plot(filtered_data)
    plt.title(f'Filtered Signal (Bandpass: {lowcut}Hz - {highcut}Hz)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot the FFT of the filtered signal
    plt.subplot(3, 1, 3)
    plt.plot(freq_fft, amplitude)
    plt.title('FFT of Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
else:
    print("The filter was applied correctly, but plotting is only implemented for 1D data.")
