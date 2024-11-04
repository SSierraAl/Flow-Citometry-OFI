import numpy as np
import os
from scipy.io import wavfile

def convert_numpy_to_wav(input_folder, output_folder, sample_rate=44100):
    """
    Convierte todos los archivos NumPy en una carpeta a archivos WAV.

    Parámetros:
        input_folder (str): La ruta de la carpeta que contiene los archivos NumPy.
        output_folder (str): La ruta de la carpeta donde se guardarán los archivos WAV.
        sample_rate (int): La tasa de muestreo para los archivos WAV.
    """
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener todos los archivos NumPy en la carpeta de entrada
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            # Cargar el archivo NumPy
            file_path = os.path.join(input_folder, file_name)
            data = np.load(file_path)

            # Convertir el nombre del archivo a .wav
            wav_file_name = file_name.replace('.npy', '.wav')
            wav_file_path = os.path.join(output_folder, wav_file_name)

            # Escribir el archivo WAV
            wavfile.write(wav_file_path, sample_rate, data)

            print(f"Archivo convertido y guardado: {wav_file_path}")

# Uso del código
input_folder = 'D:/particles/10um_DB'
output_folder = 'D:/particles/10um_WAV'
convert_numpy_to_wav(input_folder, output_folder, sample_rate=44100)