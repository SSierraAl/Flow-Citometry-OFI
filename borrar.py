import numpy as np
from scipy.io import wavfile
import os

# Directorio de entrada con archivos .wav
input_folder = 'D:/particles/10um_WAV'
# Directorio de salida para los archivos .npy
output_folder = 'D:/particles/WAV_10um_to_numpy'
# Crea la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Itera sobre todos los archivos en la carpeta de entrada
for filename in os.listdir(input_folder):
    # Verifica si el archivo tiene extensión .wav
    if filename.endswith('.wav'):
        # Construye la ruta completa del archivo
        wav_path = os.path.join(input_folder, filename)
        # Lee el archivo .wav
        sample_rate, data = wavfile.read(wav_path)

        # Crea el nombre de archivo .npy usando el nombre base del archivo .wav
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(output_folder, npy_filename)

        # Guarda los datos como un archivo .npy
        np.save(npy_path, data)

        print(f"Convertido: {filename} a {npy_filename}")

print("Conversión completada para todos los archivos .wav.")