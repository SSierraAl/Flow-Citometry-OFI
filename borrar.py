import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Función para diseñar el filtro pasabanda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Función para aplicar el filtro pasabanda
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Cargar el archivo .npy
data = np.load('.\Temporal_Files\Particles\HFocusing_0_2.npy')

# Parámetros del filtro
lowcut = 6500  # Frecuencia de corte inferior (Hz)
highcut = 50000  # Frecuencia de corte superior (Hz)
fs = 2000000  # Frecuencia de muestreo en Hz (ajusta esto a la frecuencia de muestreo de tus datos)

# Aplicar el filtro pasabanda
filtered_data = bandpass_filter(data, lowcut, highcut, fs, order=5)

# Verifica si el array es 1D para graficar
if data.ndim == 1:
    # Gráfico de la señal original
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title('Señal original')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')

    # Gráfico de la señal filtrada
    plt.subplot(2, 1, 2)
    plt.plot(filtered_data)
    plt.title(f'Señal filtrada (Pasabanda: {lowcut}Hz - {highcut}Hz)')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')

    plt.tight_layout()
    plt.show()
else:
    print("El filtro se aplicó correctamente, pero el gráfico solo está implementado para datos 1D.")
