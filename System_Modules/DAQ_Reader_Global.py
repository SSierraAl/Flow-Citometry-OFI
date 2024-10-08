
"""

Module to update the data and manage the DAQ connections

"""
###########################################################################################################################
################################################# Libraries ###############################################################
###########################################################################################################################

import numpy as np
import System_Modules.Thread_DAQ_single as tds
import pandas as pd
###########################################################################################################################

# Global holder for main_window, so other functions have access to it.
# This change was done so TAB_Scanning.py has access to the functions.
# A better way could be to define Set_DAQ_Functions() as a class
main_window = None 

def Set_DAQ_Functions(self):
    """Connects the interface buttons to the DAQ functions.
    NOTE: this used to contain all the functions, but these have now been moved outside of the Set_DAQ_Functions.
    NOTE 2: the main_window = self designation is crucial for proper functioning of those external functions.
    NOTE 3: scan_continuous_x now automatically starts the DAQ if it is not started yet, so no button has to be connected
    to the Init_DAQ_Connection function if using this function.
    """
    # Define main_window for use by other functions
    global main_window
    main_window =  self
    
    # Initialize the DAQ parameters
    Update_DAQ_Params_algorithm()
               
    # Connect interface buttons to functions#    
    #Update DAQ Parameters in TAB DAQ
    self.ui.load_pages.DAQ_connect_but.clicked.connect(Init_DAQ_Connection_algorithm)
    
    #Update DAQ Parameters in TAB Calib
    #self.ui.load_pages.continuous_scanX_but.clicked.connect(Init_DAQ_Connection_algorithm)
    self.ui.load_pages.Step_Step_but.clicked.connect(Init_DAQ_Connection_algorithm)
    self.ui.load_pages.Vel_Start_Calib.clicked.connect(Init_DAQ_Connection_algorithm)
    
    #Stop DAQ
    self.ui.load_pages.Stop_x_but.clicked.connect(Stop_DAQ_algorithm) #Stop X button
    self.ui.load_pages.Stop_Y_but.clicked.connect(Stop_DAQ_algorithm) 
    self.ui.load_pages.Stop_DAQ_but.clicked.connect(Stop_DAQ_algorithm) 

def Update_DAQ_Params_algorithm():
    """Updates the DAQ parameters from the user interface and general values
    """
    #global main_window
    main_window.Laser_Frequency = float(main_window.ui.load_pages.lineEdit_Laser.text())
    main_window.fileSave=1
    main_window.number_of_samples=int(main_window.ui.load_pages.lineEdit_number_samples.text())
    main_window.DAQ_Device="Dev1/ai0" #OLD SETUP
    #main_window.DAQ_Device= "Dev3/ai0"
    
    # Band Pass Filter Params
    main_window.order_filter=4
    main_window.low_freq_filter=float(main_window.ui.load_pages.lineEdit_Low_Freq.text())
    main_window.high_freq_filter=float(main_window.ui.load_pages.lineEdit_High_Freq.text())
    
    #RMS average
    main_window.CounterAvg=0
    main_window.VectorsAvg=int(main_window.ui.load_pages.lineEdit_Avg_FFT.text()) # Graph hold peaks
    main_window.Freq_Data=pd.DataFrame()
    
    #Save Data
    main_window.CounterPeaks=0
    main_window.exten=".npy"
    main_window.number_File=0

def Init_DAQ_Connection_algorithm():
    """Starts DAQ thread for taking measurements
    """
    Update_DAQ_Params_algorithm()
    # Creation of the thread
    print("------ DAQ Thread Init -------")
    main_window.threadDAQ = tds.DAQData(main_window.fileSave, main_window.Laser_Frequency, main_window.number_of_samples, main_window.DAQ_Device,main_window.fileSave)
    main_window.DAQ_Data= main_window.threadDAQ.DAQ_Data
    main_window.DAQ_X_Axis=main_window.threadDAQ.DAQ_X_Axis
    main_window.DAQ_X_Axis=np.array(main_window.DAQ_X_Axis)*1000/(main_window.Laser_Frequency)
    print("------ DAQ Reading Mode: On -------")

# Function placed outside of Set_DAQ_Functions() so it can be used by TAB_Scanning.py
def Stop_DAQ_algorithm():
    '''Stops the DAQ, so it can be started again by the next scan'''
    try:
        main_window.threadDAQ.deleteDAQ()
        delattr(main_window, "threadDAQ") #WARNING: This is required, as the attribute is not deleted otherwise
        # In TAB_Scanning, scan_continous_x, it is checked if there is an attribute named ThreadDAQ to check if a new thread can be started
        # Without deleting the attribute, it still shows that the attribute exists, even though the .deleteDAQ() function has been performed.
        print("------ DAQ Reading Mode: Off -------")
    except:
        print("INFO: DAQ is already stopped")