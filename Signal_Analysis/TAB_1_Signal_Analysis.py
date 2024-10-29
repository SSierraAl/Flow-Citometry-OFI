from bokeh.models import Select, NumberFormatter,NumeralTickFormatter,Button, ColumnDataSource, TextInput, DataTable, TableColumn, Div, NumeralTickFormatter, HoverTool, BoxAnnotation, RangeTool
from bokeh.plotting import figure, curdoc
from bokeh.layouts import layout, column
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def Set_Reader_Data(path, parameters):
    path_folder = path
    Adq_Freq = int(parameters['acquisition_frequency'])
    order_filt = int(parameters['bandpass_filter_order'])
    Low_Freq = int(parameters['band_low_filter_frequency'])
    High_Freq = int(parameters['band_high_filter_frequency'])

    # Load Data and Initial Setup
    Original_Dataframe = Load_New_Data(path_folder, True)
    columnas = Original_Dataframe.columns.tolist()
    Counter_Column = 0
    selected_column = columnas[Counter_Column]
    data_Y = Original_Dataframe[selected_column]
    data_X = Original_Dataframe.index / Adq_Freq * len(data_Y)
    
    # ColumnDataSource Setup
    def create_source(data_X, data_Y):
        return ColumnDataSource(data=dict(data_x_axis=data_X, daya_y_axis=data_Y))

    source_original_data = create_source(data_X, data_Y)
    source_filtered_data = create_source(data_X, data_Y)
    ampFFT1, freqfft1, _ = FFT_calc(data_Y, Adq_Freq)

    # FFT sources
    source_original_FFT = create_source(freqfft1, ampFFT1)
    source_filtered_FFT = create_source(freqfft1, ampFFT1)
    source_original_section_FFT = create_source(freqfft1, ampFFT1)
    source_filtered_section_FFT1 = create_source(freqfft1, ampFFT1)
    source_filtered_section_FFT2 = create_source(freqfft1, ampFFT1)

    # Widget Creation
    scroll_menu = Select(title="Select a column:", options=columnas)
    button_refresh = Button(label="Refresh")
    button_next = Button(label="Next")
    button_section = Button(label="Section")

    # Plot Creation Function to Avoid Redundancy
    def create_figure(title, x_axis, y_axis, width, height, source, color="blue", hover=True):
        p = figure(title=title, x_axis_label=x_axis, y_axis_label=y_axis, width=width, height=height)
        line = p.line('data_x_axis', 'daya_y_axis', source=source, line_width=2, line_color=color)
        if hover:
            p.add_tools(HoverTool(tooltips=[(x_axis, "@data_x_axis"), (y_axis, "@daya_y_axis")]))
        p.xaxis.formatter = NumeralTickFormatter(format="0a")
        return p, line

    # Figures with glyphs
    FilterFFT, FilterFFT1 = create_figure('Global FFT Filtered', '[Hz]', 'Amplitude', 700, 250, source_filtered_FFT, color='orange')
    Section_Particle, line_Particle = create_figure('Section Particle', 'Time', 'Amplitude', 700, 250, source_original_data)
    Section_Particle_Filtered, line_Particle_Filtered = create_figure('Section Particle Filtered', 'Time', 'Amplitude', 700, 250, source_filtered_data, color='orange')
    filterImpact, filterImpact1 = create_figure('Filter vs Original', '[ms]', '[V]', 1400, 250, source_original_data)
    filterImpact2 = filterImpact.line('data_x_axis', 'daya_y_axis', source=source_filtered_data, legend_label='Filtered', line_color='orange')
    Section_Particle_Filtered.x_range=Section_Particle.x_range
    # Range Tool Setup
    range_overlay = BoxAnnotation(fill_color="navy", fill_alpha=0.2)
    range_tool = RangeTool(x_range=Section_Particle.x_range, overlay=range_overlay)
    #range_tool = RangeTool(x_range=Section_Particle_Filtered.x_range, overlay=range_overlay)
    filterImpact.add_tools(range_tool)
    
    # Section FFT Figures
    SectionFFT, Section_FFT1 = create_figure('Section FFT', '[Hz]', 'Amplitude', 700, 250, source_original_section_FFT)
    Section_FilteredFFT, Section_FilterFFT1 = create_figure('Section FFT Filtered', '[Hz]', 'Amplitude', 700, 250, source_filtered_section_FFT1, color='blue')
    Section_FilterFFT2 = Section_FilteredFFT.line('data_x_axis', 'daya_y_axis', source=source_filtered_section_FFT2, line_width=2, line_color='orange')

    # Helper Function to Update Data Sources
    def update_data_sources(data_Y):
        
        Adq_Freq = int(parameters['acquisition_frequency'])
        order_filt = int(parameters['bandpass_filter_order'])
        Low_Freq = int(parameters['band_low_filter_frequency'])
        High_Freq = int(parameters['band_high_filter_frequency'])
        # Remove Mean and Apply Filter
        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, Low_Freq, High_Freq, order_filt, Adq_Freq)
        y_filtrada = y_Filtrada - y_Filtrada.mean()
        
        # Update Sources for Time and Frequency Domains
        filterImpact1.data_source.data = dict(data_x_axis=data_X, daya_y_axis=data_Y_NoOffset)
        filterImpact2.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_filtrada)
        ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)
        FilterFFT1.data_source.data = dict(data_x_axis=freqfft1, daya_y_axis=ampFFT1)

    # Function to Update Section Data Based on Range Selection
    def update_section_data():
        data_Y=Original_Dataframe[scroll_menu.value]
        start, end = Section_Particle.x_range.start, Section_Particle.x_range.end
        visible_indices = [i for i, x in enumerate(data_X) if start <= x <= end]
        visible_x = data_X[visible_indices]
        visible_y = data_Y[visible_indices]
        
        # Update Section Particle Filtered data
        y_Filtrada = butter_bandpass_filter(visible_y - np.mean(visible_y), Low_Freq, High_Freq, order_filt, Adq_Freq)
        #line_Particle_Filtered.data_source.data = dict(data_x_axis=visible_x, daya_y_axis=y_Filtrada)
        
        # Update FFT for Section FFT Figures
        amp_section_FFT, freq_section_FFT, _ = FFT_calc(visible_y, Adq_Freq)
        Section_FFT1.data_source.data = dict(data_x_axis=freq_section_FFT, daya_y_axis=amp_section_FFT)
        
        amp_filtered_FFT, freq_filtered_FFT, _ = FFT_calc(y_Filtrada, Adq_Freq)
        Section_FilterFFT1.data_source.data = dict(data_x_axis=freq_filtered_FFT, daya_y_axis=amp_filtered_FFT)

        # Optional Gaussian Curve Fitting
        x_data = np.linspace(0, len(amp_filtered_FFT), len(amp_filtered_FFT))
        y_data = amp_filtered_FFT - np.mean(amp_filtered_FFT)
        initial_params = [np.max(y_data) * 100, np.argmax(y_data), 1.0]
        optimized_params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_params)
        amplitude, mean, stddev = optimized_params
        y_curve = gaussian(x_data, amplitude, mean, stddev)
        Section_FilterFFT2.data_source.data = dict(data_x_axis=freq_filtered_FFT, daya_y_axis=y_curve)


    # Function to Move to the Next Column
    def next_column():
        nonlocal Counter_Column
        Counter_Column = (Counter_Column + 1) % len(columnas)  # Cycle back to start if at the end
        selected_column = columnas[Counter_Column]
        scroll_menu.value = selected_column  # Update dropdown selection
        update_data_sources(Original_Dataframe[scroll_menu.value])  # Refresh graph data


    # Button Actions
    button_refresh.on_click(lambda: update_data_sources(Original_Dataframe[scroll_menu.value]))
    button_section.on_click(update_section_data)
    button_next.on_click(next_column)

    # Layout Setup
    lay = layout([
        [scroll_menu, button_refresh, button_next,button_section],
        [FilterFFT,Section_FilteredFFT], [filterImpact],
        [Section_Particle, Section_Particle_Filtered]
    ])

    return lay