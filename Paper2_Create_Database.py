from Support_Functions import *
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt,spectrogram, find_peaks


global counter_counter

def Get_valuable_section(data_Y,dst_folder, filename):
    
    global counter_counter
    #Load New Column
    global Counter_Column
    Counter_Column=1

    Adq_Freq=2000000
    Time_Captured_ms=16384

    Particle_Params = {
        "laser_lambda": 1.55e-6,
        "angle": 22,
        "beam_spot_size": 90e-6,
        }
    
    data_Y = pd.Series(data_Y)

    print("-------------- ahhhhhhhhhhhhhh -----------")
    print(len(data_Y))

    data_X_max=Time_Captured_ms * 1000 / Adq_Freq
    data_X=np.linspace(0, data_X_max, Time_Captured_ms)
    #Remove Mean
    data_Y_NoOffset=data_Y
    data_Y_NoOffset = data_Y_NoOffset - data_Y_NoOffset.mean()
    y_Filtrada= butter_bandpass_filter(data_Y_NoOffset, 7000, 100000, 4, 2000000)
    ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)

    #Spectogram #####################################################################################

    #find number of peaks
    std_dev = 20  # Desviación estándar para el filtro gaussiano
    cut_index=int(len(ampFFT1)/10)
    new_FFT=ampFFT1[0:cut_index]
    filtered_fft = gaussian_filter1d(new_FFT, sigma=std_dev)
    #cut at 100Hz
    
    # Calcular la amplitud media de la señal suavizada
    #mean_amplitude = (max(filtered_fft)/3)*2
    mean_amplitude = (max(filtered_fft)/2)
    peaks, properties = find_peaks(filtered_fft, height=mean_amplitude)
    peak_amplitudes = properties["peak_heights"]
    # Ordenar los picos por su amplitud de forma descendente
    sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Agrega [::-1] para orden descendente
    sorted_peaks = peaks[sorted_indices]
    # Filtrar los picos para asegurar un espaciado mínimo de 12 kHz
    min_distance = 12000  # Espaciado mínimo en Hz
    valid_peaks = []
    for peak in sorted_peaks:
        # Comprobar espaciado con respecto a cada pico ya en valid_peaks
        if all(abs(freqfft1[peak] - freqfft1[vp]) >= min_distance for vp in valid_peaks):
            valid_peaks.append(peak)

    print(filename)    
    t, f_new, Sxx_new=find_spectogram(y_Filtrada, Adq_Freq,ampFFT1, freqfft1, Particle_Params)

    general_mean=(np.mean(Sxx_new, axis=0))
    #Reduce limit for sleection selection
    #mean_spectogram= np.mean(Sxx_new[Sxx_new < mean_spectogram],axis=0)
    t = np.linspace(0, max(data_X), len(general_mean))
    valid_zones, anomalies = classify_zones(general_mean, [5,10])

    if anomalies: 
        for aa in anomalies:
            aa=list(aa)
            if aa !=(0,0):
                if aa[0]<3:
                    aa[0]=3
                if aa[1]+3>=len(t):
                    aa[1]=-4
                # Add Span for start
                highlight_mask = (data_X >= t[aa[0]-3]) & (data_X <= t[aa[1]+3])

                highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                valid_P_x=data_X[highlight_mask]
                valid_P_y=y_Filtrada[highlight_mask]

                x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                # Step 2: Scale to -1 to 1
                valid_P_y2 = 2 * valid_P_y2 - 1
                valid_P_y2=valid_P_y-np.mean(valid_P_y)
                # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                valid_P_y2 = np.abs(hilbert(valid_P_y))
                initial_params = [np.max(valid_P_y2)*10, np.argmax(valid_P_y2), 1.0]
                try:
                    optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                    # Obtener los parámetros optimizados
                    amplitude, mean, stddev = optimized_params
                    amplitude = amplitude/ np.max((amplitude))
                    amplitude=amplitude*max(valid_P_y)
                    y_curve = gaussian(x_data, amplitude, mean, stddev)
                    if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                        
                        ####### Save  Valid_P_Y
                        # Save the file to the new destination
                        save_path = os.path.join(dst_folder, filename+str(counter_counter))
                        np.save(save_path, data_Y[highlight_mask])
                        print(f"File {filename} processed and saved to {dst_folder}")
                        counter_counter=counter_counter+1
                        #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="P_anomaly_fit")
                        #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="purple", legend_label="Gaussian_Fit")
                except:
                    pass

    if valid_zones: 
        for vv in valid_zones:
            vv=list(vv)
            if vv != (0,0):
                if vv[0]<3:
                    vv[0]=3
                if vv[1]+3>=len(t):
                    vv[1]=-4
                # Add Span for start
                highlight_mask = (data_X >= t[vv[0]-3]) & (data_X <= t[vv[1]+3])

                highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                valid_P_x=data_X[highlight_mask]
                valid_P_y=y_Filtrada[highlight_mask]
                #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                # Step 2: Scale to -1 to 1
                valid_P_y2 = 2 * valid_P_y2 - 1
                valid_P_y2=valid_P_y-np.mean(valid_P_y)
                # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                valid_P_y2 = np.abs(hilbert(valid_P_y))
                initial_params = [np.max(valid_P_y2)*10, np.argmax(valid_P_y2), 1.0]
                try:
                    optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                    # Obtener los parámetros optimizados
                    amplitude, mean, stddev = optimized_params
                    amplitude = amplitude/ np.max((amplitude))
                    amplitude=amplitude*max(valid_P_y)
                    y_curve = gaussian(x_data, amplitude, mean, stddev)
                    if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                        #Save valid_P_Y
                        
                        save_path = os.path.join(dst_folder, filename+str(counter_counter))
                        np.save(save_path, data_Y[highlight_mask])
                        print(f"File {filename} processed and saved to {dst_folder}")
                        counter_counter=counter_counter+1
                        #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                        #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="green", legend_label="Gaussian_Fit")
                    else:
                        #Save valid_P_Y
                        pass
                        #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="yellow", legend_label="P_No_Fit")
                        #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="yellow", legend_label="Gaussian_Fit")
                except:
                    pass
                    #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="yellow", legend_label="P_No_Fit")

    #########################################################
    ##########################################################

    if len(valid_peaks)>1:

        two_peak=int(freqfft1[valid_peaks][1])
        min_lim=(two_peak)-12000
        if min_lim<7000:
            min_lim=7000
        max_lim=two_peak+12000
        print('filter limits')
        print(min_lim)    
        print(max_lim)     
        data_Y_NoOffset=data_Y
        data_Y_NoOffset = data_Y_NoOffset - data_Y_NoOffset.mean()
        y_Filtrada= butter_bandpass_filter(data_Y_NoOffset, min_lim, max_lim, 4, Adq_Freq)
        ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)
        #New Specdtogram
        t, f_new, Sxx_new=find_spectogram(y_Filtrada, Adq_Freq,ampFFT1, freqfft1, Particle_Params)
        general_mean=(np.mean(Sxx_new, axis=0))
        mean_spectogram=np.mean(general_mean)
        #Reduce limit for sleection selection
        #mean_spectogram= np.mean(Sxx_new[Sxx_new < mean_spectogram],axis=0)
        t = np.linspace(0, max(data_X), len(general_mean))
        #color_mapper = LinearColorMapper(palette=Viridis256, low=(np.min(Sxx_new)), high=(np.max(Sxx_new)))

        valid_zones, anomalies = classify_zones(general_mean, [5,10])
        print(valid_zones)
        print(anomalies)
        if anomalies: 
            for aa in anomalies:
                aa=list(aa)
                if aa !=(0,0):
                    if aa[0]<3:
                        aa[0]=3
                    if aa[1]+3>=len(t):
                        aa[1]=-4
                    # Add Span for start
                    highlight_mask = (data_X >= t[aa[0]-3]) & (data_X <= t[aa[1]+3])

                    highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                    valid_P_x=data_X[highlight_mask]
                    valid_P_y=y_Filtrada[highlight_mask]

                    x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                    valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                    # Step 2: Scale to -1 to 1
                    valid_P_y2 = 2 * valid_P_y2 - 1
                    valid_P_y2=valid_P_y-np.mean(valid_P_y)
                    # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                    valid_P_y2 = np.abs(hilbert(valid_P_y))
                    initial_params = [np.max(valid_P_y2)*10, np.argmax(valid_P_y2), 1.0]
                    try:
                        optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                        # Obtener los parámetros optimizados
                        amplitude, mean, stddev = optimized_params
                        amplitude = amplitude/ np.max((amplitude))
                        amplitude=amplitude*max(valid_P_y)
                        y_curve = gaussian(x_data, amplitude, mean, stddev)
                        if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                            #Save Valid_P_Y
                            save_path = os.path.join(dst_folder, filename+str(counter_counter))
                            np.save(save_path, data_Y[highlight_mask])
                            print(f"File {filename} processed and saved to {dst_folder}")
                            counter_counter=counter_counter+1

                            #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="P_anomaly_fit")
                            #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="purple", legend_label="Gaussian_Fit")
                    except:
                        pass
        if valid_zones: 
            for vv in valid_zones:
                vv=list(vv)
                if vv != (0,0):
                    if vv[0]<3:
                        vv[0]=3
                    if vv[1]+3>=len(t):
                        vv[1]=-4
                    # Add Span for start
                    highlight_mask = (data_X >= t[vv[0]-3]) & (data_X <= t[vv[1]+3])

                    highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                    valid_P_x=data_X[highlight_mask]
                    valid_P_y=y_Filtrada[highlight_mask]
                    #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                    x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                    valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                    # Step 2: Scale to -1 to 1
                    valid_P_y2 = 2 * valid_P_y2 - 1
                    valid_P_y2=valid_P_y-np.mean(valid_P_y)
                    # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                    valid_P_y2 = np.abs(hilbert(valid_P_y))
                    initial_params = [np.max(valid_P_y2)*10, np.argmax(valid_P_y2), 1.0]
                    try:
                        optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                        # Obtener los parámetros optimizados
                        amplitude, mean, stddev = optimized_params
                        amplitude = amplitude/ np.max((amplitude))
                        amplitude=amplitude*max(valid_P_y)
                        y_curve = gaussian(x_data, amplitude, mean, stddev)
                        if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                            # Save Valid_P_Y
                            save_path = os.path.join(dst_folder, filename+str(counter_counter))
                            np.save(save_path, data_Y[highlight_mask])
                            print(f"File {filename} processed and saved to {dst_folder}")
                            counter_counter=counter_counter+1
                            #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                            #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="green", legend_label="Gaussian_Fit")
                    except:
                        pass


def process_and_save_files(path_folder,source_folders, destination_folders):

    global counter_counter
    counter_counter=0
    if len(source_folders) != len(destination_folders):
        raise ValueError("Source and destination folders lists must have the same length.")
    
    src_folder_new=[]
    for i in source_folders:
        src_folder_new.append(path_folder+i)
    source_folders=src_folder_new

    destination_folders_new=[]
    for i in destination_folders:
        destination_folders_new.append(path_folder+i)
    destination_folders=destination_folders_new


    for src_folder, dst_folder in zip(source_folders, destination_folders):
        # Ensure destination folder exists
        os.makedirs(dst_folder, exist_ok=True)
        
        # List all files in the source folder
        for filename in os.listdir(src_folder):
            if filename.endswith('.npy'):
                file_path = os.path.join(src_folder, filename)
                
                # Load the numpy file
                data = np.load(file_path)

                Get_valuable_section(data,dst_folder,filename)


path_folder="D:/particles/"

Name_Folder=['signal_10um_CAM',
             'signal_4um_CAM']

Final_Folder=['10um_DB_RAW',
             '4um_DB_RAW']

process_and_save_files(path_folder,Name_Folder, Final_Folder)