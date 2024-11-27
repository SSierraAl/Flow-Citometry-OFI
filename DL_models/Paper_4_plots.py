

"""

Codigo para plotear histogramas de mis particulas y mis datos!!

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




from bokeh.layouts import layout, column
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import os
from scipy.signal import find_peaks, hilbert

# Parámetros de configuración
Adq_Freq = 2000000
order_filt = 4
Low_Freq = 7000
High_Freq = 80000

# Ruta de las carpetas
#paths = ['D:/particles/DB_10um', 'D:/particles/WAV_4um_to_numpy - copia', 'D:/particles/WAV_2um_to_numpy']
paths = ['D:/particles/Paper_DATA_2um_augmented', 'D:/particles/Paper_DATA_4um_augmented', 'D:/particles/Paper_DATA_10um_augmented']

#   Load multiple files
################################################################################
def  Load_New_Data(filepath, newdata):
    if newdata==True:
        folder_path = filepath
        file_names = sorted(os.listdir(folder_path))
        data_list = []
        del file_names[0]
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            data_list.append(data)
        #np.save('outputConcatenatedData.npy', data_list)
        df = pd.DataFrame(data_list)

        Data_Frame= df.transpose()
        num_cols = Data_Frame.shape[1]
        interval = 10

        for i in range(num_cols):
            pos = i // interval
            exp = i % interval
            col_name = f'Pos{pos}_exp{exp}'
            col_name=file_names[i]
            Data_Frame.rename(columns={i: col_name}, inplace=True)
        Data_Frame.to_csv('DataframewithOrder.csv', index=False)

    else:
        Data_Frame = pd.read_csv('./DataframewithOrder.csv')

    return Data_Frame


def count_peaks_with_gaussian(signal, data_X, main_freq):
    # Step 1: Preprocess the signal by setting all values less than zero to zero
    processed_signal = np.maximum(signal, 0)
    valid_P_y2 = butter_bandpass_filter(processed_signal, main_freq-5000, main_freq+5000, order_filt, Adq_Freq)
    # Step 2: Perform Gaussian fitting on the envelope of the signal
    #valid_P_y2 = np.abs(hilbert(valid_P_y2))  # Hilbert transform for envelope


    valid_P_y2_count=np.maximum(valid_P_y2, 0)


    try:
        # Fit Gaussian to the signal envelope
        optimized_params, _ = curve_fit(gaussian, data_X, valid_P_y2_count, maxfev=10)
        amplitude, mean, stddev = optimized_params
        amplitude_normalized = (amplitude / np.max(amplitude)) * max(signal)
        stddev=stddev*2/3
        y_curve = gaussian(data_X, amplitude_normalized, mean, stddev)


    except Exception as e:
        print("Gaussian fit failed:", e)
        return 0, 0  # Return zero peaks if the fit fails

    # Calcular el umbral como la mitad de la altura máxima de la gaussiana
    threshold = y_curve.max() / 4
    # Encuentra los índices donde la curva cruza el umbral
    above_threshold = y_curve > threshold
    crossing_indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]

    # Verificamos que hay dos cruces
    if len(crossing_indices) >= 2:
        # Primer y segundo cruce del umbral
        first_crossing = crossing_indices[0]
        second_crossing = crossing_indices[1]
        
        # Rango delimitado para el conteo de picos
        valid_range = valid_P_y2_count[first_crossing:second_crossing + 1]

        # Detectar y contar los picos en el rango seleccionado
        peak_count = 0
        for i in range(1, len(valid_range) - 1):
            # Condición de un pico: el valor actual es mayor a cero y es un punto de subida
            if valid_range[i-1] <= 0.00000 and valid_range[i] > 0.0000000:
                peak_count += 1

        # Calcular el ancho en milisegundos entre los cruces
        width_points = ((len(valid_range)/Adq_Freq) * 1000)
    else:
        peak_count = 0
        width_points = 0


    return peak_count, width_points


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))


# Función para cargar y procesar datos de cada carpeta
def load_and_process_data(path):
    data = Load_New_Data(path, True)  # Cargar datos
    resultados = pd.DataFrame()  # Inicializamos el DataFrame para guardar resultados

    for col in data.columns:
        data_Y = data[col].fillna(0)
        
        # Preprocesamiento y filtrado de señal
        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, Low_Freq, High_Freq, order_filt, Adq_Freq)
        ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)

        # Conteo de picos y paso usando la función auxiliar
        num_bursts, passage = count_peaks_with_gaussian(y_Filtrada, data.index / Adq_Freq * len(data_Y), freqfft1[np.argmax(ampFFT1)])
        
        if max(y_Filtrada)<1.05:
            pass

        if path==paths[2]:
            if passage<0.9:
                # Añadir los resultados de la columna al DataFrame de resultados
                resultados = pd.concat([resultados, pd.DataFrame({
                    'name': [col],
                    'val_Freq': [freqfft1[np.argmax(ampFFT1)]/1000],
                    'amp_Freq': [max(ampFFT1)],
                    'amp_Time': [max(y_Filtrada)],
                    'num_burst': [num_bursts],
                    'passage': [passage]
                })], ignore_index=True)
                

        else:

            # Añadir los resultados de la columna al DataFrame de resultados
            resultados = pd.concat([resultados, pd.DataFrame({
                'name': [col],
                'val_Freq': [freqfft1[np.argmax(ampFFT1)]/1000],
                'amp_Freq': [max(ampFFT1)],
                'amp_Time': [max(y_Filtrada)],
                'num_burst': [num_bursts],
                'passage': [passage]
            })], ignore_index=True)

    return resultados

# Cargar datos de ambas carpetas

data_10um = load_and_process_data(paths[0])
data_4um = load_and_process_data(paths[1])
data_2um = load_and_process_data(paths[2])

# Lista de columnas para graficar
columns_to_plot = data_10um.columns[1:]  # Excluimos 'name' para los histogramas

# Determina el tamaño de la cuadrícula (ej. 2x3 si hay 6 columnas para graficar)
num_columns = len(columns_to_plot)
num_rows = (num_columns + 2) // 3  # Dividir en 3 columnas, ajustando las filas según el número total de columnas

columna_a_graficar= 'val_Freq'

# Verificar si la columna está en los datos antes de graficar
if columna_a_graficar in data_10um.columns:
    # Definir los colores accesibles
    colors_d_friendly = ['#0072B2', '#E69F00', '#CC79A7','#009E73']  # Azul, Naranja, Púrpura

    plt.figure(figsize=(10, 6))

    # Configurar la fuente y tamaño de letra para Times New Roman y agrandado
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 16,  # Tamaño de fuente general
        'axes.titlesize': 20,  # Tamaño del título del gráfico
        'axes.labelsize': 30,  # Tamaño de etiquetas de los ejes
        'xtick.labelsize': 28,  # Tamaño de las etiquetas del eje x
        'ytick.labelsize': 28,  # Tamaño de las etiquetas del eje y
        'legend.fontsize': 28   # Tamaño de la leyenda
    })

    # Crear histogramas con borde sutil, colores accesibles y cuadrícula
    plt.hist(data_10um[columna_a_graficar], bins=15, alpha=0.6, label='10um', edgecolor='gray', linewidth=0.8, color=colors_d_friendly[0])
    plt.hist(data_4um[columna_a_graficar], bins=15, alpha=0.6, label='4um', edgecolor='gray', linewidth=0.8, color=colors_d_friendly[1])
    plt.hist(data_2um[columna_a_graficar], bins=15, alpha=0.6, label='2um', edgecolor='gray', linewidth=0.8, color=colors_d_friendly[2])

    # Personalización del gráfico
    #plt.title(f'Amplitude (mV) {columna_a_graficar}')
    plt.xlabel(f'val_Freq')
    plt.ylabel('Events')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # Añadir cuadrícula con líneas punteadas y transparencia

    plt.show()
else:
    print(f"La columna '{columna_a_graficar}' no existe en los datos.")