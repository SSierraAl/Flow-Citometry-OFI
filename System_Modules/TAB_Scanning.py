"""TAB_Scanning.py is responsible for the movement of the Zaber and
the data processing of the laser measurements, including the flow velocity profile
"""

###########################################################################################################################
################################################# Libraries ###############################################################
###########################################################################################################################

from Support_Functions import *
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel,QVBoxLayout,QGraphicsView
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
import zaber_motion
from zaber_motion import Units
from zaber_motion.binary import Connection,CommandCode
#from zaber_motion.ascii import Connection
import  System_Modules.DAQ_Reader_Global as DAQ_Reader_Global  # Import for use of stop_daq() function
import System_Modules.scan_area_module as scan_area_module

###########################################################################################################################

#Graphic Widget

class ColorGrid(QWidget):
    def __init__(self, g_W, g_H, cell_size):
        super().__init__()
        self.g_W = g_W
        self.g_H = g_H
        self.cell_size = cell_size

        # Matrix to export CSV data
        self.CSV_Moments_M0_Matrix=np.zeros((self.g_W,self.g_H))
        self.CSV_Moments_M1_Matrix=np.zeros((self.g_W,self.g_H))
        self.CSV_Moments_Matrix=np.zeros((self.g_W,self.g_H))

        # Create PlotItem and add it to the GraphicsLayout
        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setBackgroundBrush(QColor('white'))
        self.graphicsLayout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.graphicsLayout)

        # Create PlotItem and add it to the GraphicsLayout
        self.plotItem = self.graphicsLayout.addPlot()
        self.plotItem.showGrid(x=True, y=True)
        self.plotItem.setRange(xRange=(0, self.g_W), yRange=(0, self.g_H), padding=0)
        
        # Create the ImageItem and add it to the PlotItem
        self.imageItem = pg.ImageItem()
        self.plotItem.addItem(self.imageItem)
        self.plotItem.getViewBox().setBackgroundColor('w')
        self.gridArray = np.ones((self.g_H, self.g_W, 3), dtype=np.uint8)*255

        # Configure the design
        layout = QVBoxLayout(self)
        layout.addWidget(self.graphicsView)

        self.timer_Calib_Scan = QTimer(self)
        self.timer_Calib_Scan.setInterval(1000)  # Intervalo en milisegundos
        self.timer_Calib_Scan.timeout.connect(self.updatePixel)
        # Variables for traversing pixels
        self.current_row = 0
        self.current_col = 0


    def updatePixel(self):

        # Change the colour of the current pixel
        self.gridArray[self.current_row, self.current_col] = [255, 0, 0]  # Red
        # Updating the image
        self.imageItem.setImage(self.gridArray, autoLevels=False)
        # Move to the next pixel
        self.current_col += 1
        if self.current_col >= self.g_W: # If end of row has been reached
            self.current_col = 0  # Reset column to beginning
            self.current_row += 1 # Go to next row
            if self.current_row >= self.g_H: # If last row has been reached
                self.current_row = 0  # Reset row


    def changeCellColor(self, row, col, color):
        self.gridArray[row, col] = color
        self.imageItem.setImage(self.gridArray, autoLevels=False)

    def Start_Grid_Scan(self):
        self.timer_Calib_Scan.start()

    def Stop_Grid_Scan(self):
        self.timer_Calib_Scan.stop()

    def update_white(self):
        self.gridArray = np.zeros((self.g_H, self.g_W, 3), dtype=np.uint8)*255
        self.imageItem.setImage(self.gridArray, autoLevels=False)

    ### Export CSV functions
    def updateCSV(self, row, col, Moment, M0, M1):
        self.CSV_Moments_Matrix[row, col] = Moment
        self.CSV_Moments_M0_Matrix[row, col] = M0
        self.CSV_Moments_M1_Matrix[row, col] = M1


    def exportar_Matrix_CSV(self):
        df = pd.DataFrame(self.CSV_Moments_Matrix)
        path=r'.\Temporal_Files\Scanning'
        df.to_csv(path+r'\Scanning_Moments_CSV.csv', index=True) # r to declare raw string
        df = pd.DataFrame(self.CSV_Moments_M0_Matrix)
        df.to_csv(path+r'\Scanning_Moments_M0_CSV.csv', index=True)
        df = pd.DataFrame(self.CSV_Moments_M1_Matrix)
        df.to_csv(path+r'\Scanning_Moments_M1_CSV.csv', index=True)

    ###################################################################
    ###################################################################

class Scan_functions:
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Button connections
        self.main_window.ui.load_pages.Calib_Reset_Scan_but.clicked.connect(self.reset_refresh_scan)
        self.main_window.ui.load_pages.Stop_x_but.clicked.connect(self.stop_z1) # Button for stopping X
        self.main_window.ui.load_pages.Stop_Y_but.clicked.connect(self.stop_z2)
        self.main_window.ui.load_pages.continuous_scanX_but.clicked.connect(self.start_move_x)
        self.main_window.ui.load_pages.continuous_scanY_but.clicked.connect(self.start_move_y)

    def reset_refresh_scan(self):
        '''Resets the grid for a new scan. Without this, a second grid will appear.
        '''
        # Check if color_grid_widget exists or not
        if not hasattr(self.main_window,'color_grid_widget'): # was self.main_window
            print("INFO: No grid to reset")
            return
        
        # The color_grid_widget is dynamically added, so pylint gives an error on the follow lines.
        # This error is disabled for the specific lines using the disable comments.
        self.main_window.color_grid_widget.update_white()
        self.main_window.color_grid_widget.Stop_Grid_Scan()
        self.main_window.ui.load_pages.Layout_table_Scan.removeWidget(self.main_window.color_grid_widget)
        print("INFO: Flow Velocity Profile grid reset for new scan")

    # General stop zaber function
    def stop_zaber(self, zaber):
        """Sends a stop command to a zaber

        Args:
            zaber (int): Zaber to stop. Index starts at 1. 0 is invalid.
        """
        self.main_window.Pixel_Interval.stop()
        self.main_window.Adquisit_Timer.stop()

        #Binary
        with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
            connection.generic_command(zaber, CommandCode.STOP, 1)
        
        # Ascii
        #with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
        #    connection.generic_command("stop", zaber, 0, True, 50)

    # Stop Zaber in X direction
    # Function for using the buttons properly
    # NOTE: A button connection can't provide arguments, so this intermediate function 
    # is required.
    def stop_z1(self):
        self.stop_zaber(1)

    # Stop Zaber in Y direction
    # Function for using the buttons properly
    def stop_z2(self):
        self.stop_zaber(3)        

            
    def move_to_position(self, zaber, position):
        """Moves a zaber to a designated position

        Args:
            zaber (int): Zaber to move, index starts at 1. 0 is invalid.
            position (int): position to move to in um.
        """
        with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
            #device_list = connection.detect_devices()
            print(f"INFO: Moving Zaber {zaber} to position {position} ")
            
            # UNIT CONVERSION: position = data x (Micropstep Size)
            # data to zaber = position / microstep size
            position_absolute = int(position / (0.047625)) #* 10**(-6))
            
            # Send move command to Zaber
            #Binary
            device_list = connection.detect_devices()
            device_list[zaber].move_absolute(position, Units.LENGTH_MICROMETRES)
            #Ascii
            #connection.generic_command(f"move abs {position_absolute}", zaber, 0, True, 500)
            # HOLD until Zaber finished moving
            # NOTE: this blocks the rest of the program and freezes the UI, but is required for current implementation-
            # of continuous measurements. Otherwise the rest of the program starts measurements etc. while zaber is not
            # yet in start position.
            #actual_status = connection.generic_command("get motion.busy",zaber, 0, True, 500)
            #while actual_status.status == 'BUSY':
            #    actual_status = connection.generic_command("get motion.busy",zaber, 0, True, 500)

    def get_current_position(self, zaber):
        """Gets the current position of a Zaber

        Args:
            zaber (int): Zaber number to ask position from. Index starts at 1.

        Returns:
            Current position in um
        """
        with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
            
            #Ascii
            #return_value = connection.generic_command("get pos", zaber, 0, True, 500)
            #current_position_absolute = float(return_value.data) #NOTE: .data contains the absolute position
            
            # Unit conversion: position = data × (Microstep Size)
            #current_position_um = current_position_absolute * (0.047625) #* 10**(-6))
            #print(f"INFO: Current position of Zaber {zaber}: {current_position_um} um")

            #Binary
            device_list = connection.detect_devices()
            current_position_um=device_list[zaber].get_position(Units.LENGTH_MICROMETRES)
            return current_position_um

    def start_continuous_movement(self):
        """Performs continous movement of Zaber NOTE: in X direction.
        Calculates time it takes to travel along entire distance and stops
        the zaber when this time, and thus distance, have passed.
        """
        print("INFO: starting continuous movement function")
        # TODO: There should be some way to optimize this code with less lines, but it works at the moment. 
        # It is easiest to keep the override and manual things separate, to make sure there is
        # no weird behaviour with other code.
        
        if self.main_window.edge_scan_mode is True:
            print("INFO: using algorithm override settings for continuous movement")
            self.main_window.distance = abs(self.main_window.Pos_X2_Scan_override - self.main_window.Pos_X1_Scan_override)
            self.main_window.time_to_travel = abs(self.main_window.distance / float(self.main_window.speed_override))*1000 # Milliseconds
            self.main_window.cell_size = int(self.main_window.cell_size_override) 
            self.main_window.time_timer_scan=abs((self.main_window.time_to_travel/(self.main_window.distance/self.main_window.cell_size)))
        else:
            print("INFO: using interface settings for continuous movement")
            self.main_window.distance = abs(self.main_window.Pos_X2_Scan - self.main_window.Pos_X1_Scan)
            self.main_window.time_to_travel = abs((self.main_window.distance / float(self.main_window.ui.load_pages.lineEdit_speed_ums.text())))*1000 # Milliseconds
            self.main_window.cell_size = int(self.main_window.ui.load_pages.lineEdit_Pixel_size.text()) 
            self.main_window.time_timer_scan=abs((self.main_window.time_to_travel/(self.main_window.distance/self.main_window.cell_size)))
       
        # NOTE: Debug prints for used settings
        #print(f"INFO: Speed settings: {self.main_window.speed} {float(self.main_window.ui.load_pages.lineEdit_speed_ums.text())}, {self.main_window.speed_override}")
        #print(f"INFO:  Scan settings: distance:{self.main_window.distance}; time to travel:{self.main_window.time_to_travel}; cell size: {self.main_window.cell_size}; time timer scan:{self.main_window.time_timer_scan}")
        #print(f"{self.main_window.distance}, {self.main_window.time_to_travel}, {self.main_window.cell_size},{self.main_window.time_timer_scan}")

        print("INFO: attempting Zaber move constant speed")
        try:
            # TODO:  switch the order of if statement, makes more sense to check for edge scan mode is true
            if self.main_window.edge_scan_mode is False:
                print (f"Manual speed setting: {self.main_window.speed}")
                # UNIT CONVERSION: velocity  = data × (Microstep Size) / (1.6384 s)
                # data = velocity / (Microstep Size) * (1.6384 s)
                # NOTE: data is received from or sent to Zabers
                
                speed_absolute = int(self.main_window.speed / (0.047625) * 1.6384)
                print(f"INFO: absolute speed setting: {speed_absolute}")

                print("INFO: Sending Zaber move command")

                #Binary
                if self.main_window.axis == 'x':
                    with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                        connection.generic_command(3, CommandCode.MOVE_AT_CONSTANT_SPEED, speed_absolute)
                if self.main_window.axis == 'y':
                    with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                        connection.generic_command(1, CommandCode.MOVE_AT_CONSTANT_SPEED, speed_absolute)

                #Ascii
                #with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                #    connection.generic_command(f"move vel {speed_absolute}", 1, 0, True, 500)    

            else:
                print (f"EDGE SCAN MODE: Algorithm Override speed setting: {self.main_window.speed_override}")
                
                # UNIT CONVERSION: velocity = data × (Microstep Size) / (1.6384 s)
                override_speed_absolute = int(self.main_window.speed_override / (0.047625) * 1.6384)
                print(f"INFO: absolute speed setting: {override_speed_absolute}")
                
                print(f"INFO: Sending Zaber move command in {self.main_window.axis} direction")
                if self.main_window.axis == 'x':
                    
                    #Binary
                    with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                        connection.generic_command(3, CommandCode.MOVE_AT_CONSTANT_SPEED, override_speed_absolute)
                    #Ascii
                    #with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                    #    response = connection.generic_command(f"move vel {override_speed_absolute}", 1, 0, True, 500)
                    #    print(response)
                
                elif self.main_window.axis == 'y':
                
                    #Binary
                    with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                        connection.generic_command(1, CommandCode.MOVE_AT_CONSTANT_SPEED, override_speed_absolute)
                    
                    #Ascii
                    #with Connection.open_serial_port(self.main_window.Zaber_COM) as connection:
                    #    response = connection.generic_command(f"move vel {override_speed_absolute}", 2, 0, True, 500)
                    #    print(response)
                
                else:
                    print("ERROR: invalid axis argument")
        except:
            print("ERROR: Zaber move command failed")
            self.start_continuous_movement() # retry

        self.main_window.Pixel_Interval.setInterval(self.main_window.time_timer_scan)
        self.main_window.Pixel_Interval.start()
        self.main_window.Adquisit_Timer.start()
        
    def check_position_and_start(self, Zab, reference_Zab):
        """Starts continuous movement along X if Zaber is in reference position
        Otherwise, give command to move to reference position.
        Args:
            Zab (_type_): _description_
            reference_Zab (_type_): _description_
        """
        current_position = self.get_current_position(Zab)
        
        if abs(current_position - reference_Zab) < 10:
            print("STATUS: position reached")
            self.start_continuous_movement()
        else:
            print("STATUS: continue movement")
            QTimer.singleShot(500, self.check_position_and_start(Zab,reference_Zab))

    def interpolation_color(self, oldcolor):
        """Adjusts the displayed color of flow velocity profile display 
        -(dark green, bright green etc.)
        The range of measurement value inputs (min_x, max_x) are converted to 
        the same ratio but from 0 to 255. This can then be used for colors
        
        NOTE: take care in defining these values, if all measured values are 
        lower than min_x, for example, you won't see any colors in the results.

        Args:
            oldcolor (integer): the average momentum measured by the laser

        Returns:
            newcolor (integer): the color value within a range of 0 to 255, 
            with the lowest input resulting in 255, and max input in 0.
        """
        min_x = 0 #maybe change to 17000
        max_x = 30 #maybe change to 45000
        min_y = 0
        max_y = 255
        newcolor = ((oldcolor - min_x) * (max_y - min_y) / (max_x - min_x)) + min_y
        return newcolor


    def start_move_x(self):
        self.main_window.axis = 'x'
        self.start_manual_scan()

    def start_move_y(self):
        self.main_window.axis = 'y'
        self.start_manual_scan()

    def start_manual_scan(self):
        """Starts a manual scan, makes sure the user interface settings are used.
        """
        self.main_window.override_user_settings = False
        self.main_window.edge_scan_mode = False
        #self.main_window.axis = 'x' # change this to y if y direction is required.
        self.scan_continuous()

    def scan_continuous(self):
        """Sets up the continuous scan.
        Currently has two separate modes: user settings or edge scan mode.
        Depending on the mode, it takes different input settings and applies them
        To the GUI velocity profile graph settings and then starts a scan in either x or y direction
        NOTE: the velocity profile works in both directions
        """
        print("------ Starting Continuous Scan ------")
        # Reset grid so there won't be 2 grids if there's already a grid with previous measurements
        if hasattr(self.main_window,'color_grid_widget'):
            self.reset_refresh_scan()
            
        # Determine which source of settings should be used
        # NOTE: at the moment only edge_scan_mode needs to apply specific settings
        # But more modes can be added, or other functions can just use the same ..._override variables.
        if self.main_window.edge_scan_mode is True:
            # Use algorithm override settings
            print("INFO: using algorithm override scan settings")
            self.main_window.speed = self.main_window.override_user_settings
            self.main_window.Pos_Y1_Scan = self.main_window.Pos_Y1_Scan_override
            self.main_window.Pos_Y2_Scan = self.main_window.Pos_Y2_Scan_override
            self.main_window.Pos_X1_Scan = self.main_window.Pos_X1_Scan_override
            self.main_window.Pos_X2_Scan = self.main_window.Pos_X2_Scan_override
            self.main_window.cell_size = self.main_window.cell_size_override
        else:
            # Use user interface settings
            print("INFO: using user interface scan settings")
            self.main_window.speed = float(self.main_window.ui.load_pages.lineEdit_speed_ums.text())
            self.main_window.Pos_Y1_Scan = float(self.main_window.ui.load_pages.lineEdit_y1_scan.text())
            self.main_window.Pos_Y2_Scan = float(self.main_window.ui.load_pages.lineEdit_y2_scan.text())
            self.main_window.Pos_X1_Scan = float(self.main_window.ui.load_pages.lineEdit_x1_scan.text())
            self.main_window.Pos_X2_Scan = float(self.main_window.ui.load_pages.lineEdit_x2_scan.text())
            self.main_window.cell_size = int(self.main_window.ui.load_pages.lineEdit_Pixel_size.text())

        #Update GUI width and height
        self.main_window.g_W=abs(int((self.main_window.Pos_X2_Scan-self.main_window.Pos_X1_Scan)/self.main_window.cell_size))
        self.main_window.g_H=abs(int((self.main_window.Pos_Y2_Scan-self.main_window.Pos_Y1_Scan)/self.main_window.cell_size))
        
        #Create Graph
        self.main_window.color_grid_widget = ColorGrid(self.main_window.g_H,self.main_window.g_W,self.main_window.cell_size)
        self.main_window.ui.load_pages.Layout_table_Scan.addWidget(self.main_window.color_grid_widget)
        
        # Variables_Initialization
        self.main_window.counter_Data_per_Pixel=0
        self.main_window.counter_Step_Zaber_X=0
        self.main_window.current_col=0
        self.main_window.current_row=0
        self.main_window.New_Color= [255,255,255] # [R,G,B] => WHITE

        # Average dev flow rate
        self.main_window.Moment_Dev=pd.DataFrame()
        self.main_window.Samples_to_AVG=0
        self.main_window.Samples_To_AVG_Flag=True

        # Move first to (X1,Y1)
        self.move_to_position(0,self.main_window.Pos_Y1_Scan)
        self.move_to_position(2,self.main_window.Pos_X1_Scan)
        
        # (Re)start DAQ if required.
        # NOTE: this is intentionally done after move_to_position, to make sure the DAQ
        # does not start measuring things during the movement to the starting position
        if not hasattr(self.main_window, 'threadDAQ'):
            print("INFO: starting DAQ")
            DAQ_Reader_Global.Init_DAQ_Connection_algorithm()

        # Start scan
        if self.main_window.axis == 'x':
            self.check_position_and_start(2,self.main_window.Pos_X1_Scan)
        elif self.main_window.axis == 'y':
            self.check_position_and_start(0,self.main_window.Pos_Y1_Scan)
        else:
            # PLACEHOLDER for adding Z axis if desired someday.
            print("ERROR: invalid axis argument")

    def change_pixel_daq(self):
        """Uses measurements from the DAQ, saves the relevant information and updates the flow velocity profile
        
        This function is periodically called by a timer during scans.
        
        NOTE: measurements work in both x and y direction scan, but the saving of .csv files and other reports
        is only supported for scans in x direction. Y direction might work, but is not tested.
        """
        
        # If end of line is reached
        if ((self.main_window.axis == 'x' and self.main_window.counter_Data_per_Pixel >= self.main_window.g_W)
            or (self.main_window.axis == 'y' and self.main_window.counter_Data_per_Pixel >= self.main_window.g_H)):
            self.main_window.counter_Data_per_Pixel=0
            
            # Stop the movement
            self.stop_zaber(1)
            self.stop_zaber(3)
            
            actual_pos=(self.main_window.Pos_Y1_Scan+(((self.main_window.Pos_Y2_Scan-self.main_window.Pos_Y1_Scan)/self.main_window.g_H)*self.main_window.counter_Step_Zaber_X))
            
            # Save Avg spectrum for each row to .csv, only in normal scan mode and for x direction
            # NOTE: if necessary it should work for y, but not tested and not currently required.
            if self.main_window.edge_scan_mode is False and self.main_window.axis == 'x':
                path=r'.\Temporal_Files\Scanning'
                self.main_window.Data_Spectrum_Array.to_csv(path+r'\Scanning_Avg_Spectrum'+str(actual_pos)+'.csv', index=True)
            self.main_window.Data_Spectrum_Array=pd.DataFrame()
        
            # Reset positions
            self.main_window.counter_Step_Zaber_X+=1
            
            # Determine next position, depending on axis.
            # NOTE: the name implies it is only for x, but it is used for both x and y movements.
            if self.main_window.axis == 'x':
                new_x_pos=  (self.main_window.Pos_Y1_Scan
                            +(((self.main_window.Pos_Y2_Scan-self.main_window.Pos_Y1_Scan)/self.main_window.g_H)
                            *self.main_window.counter_Step_Zaber_X))
            elif self.main_window.axis == 'y':
                new_x_pos=  (self.main_window.Pos_X1_Scan
                            +(((self.main_window.Pos_X2_Scan-self.main_window.Pos_X1_Scan)/self.main_window.g_W)
                            *self.main_window.counter_Step_Zaber_X))
            else:
                print("ERROR: invalid axis: {self.main_window.axis}")
            
            # if final row has been finished, meaning all rows have been scanned
            # OR if edge_scan_mode, because it only needs to scan 1 row.
            if ((self.main_window.axis == 'x' and new_x_pos>=self.main_window.Pos_Y2_Scan) 
                or (self.main_window.axis == 'y' and new_x_pos>=self.main_window.Pos_X2_Scan)
                or self.main_window.edge_scan_mode is True):
                
                if self.main_window.edge_scan_mode is True:
                    # If end of scan range has been reached, it means that no edge has been found
                    print("EDGE SCAN MODE: No edge found")
                                
                self.main_window.Adquisit_Timer.stop()

                # Save x direction scan moment values to .csv
                if self.main_window.edge_scan_mode is False and self.main_window.axis == 'x':
                    self.main_window.color_grid_widget.exportar_Matrix_CSV()
                    path=r'.\Temporal_Files\Scanning'
                    self.main_window.Moment_Dev.to_csv(path+r'\Scanning_Moments_Dev.csv', index=True)
                
                print('------  Scan finished --------')
                print('NOTE: Do not forget to save .csv files to other directory before starting next scan')
                
                # Stop DAQ if it is already running, can cause crash otherwise
                if hasattr (self.main_window, 'threadDAQ'):
                    DAQ_Reader_Global.Stop_DAQ_algorithm()
            
            else:
                # Move to next position and start new movement
                if self.main_window.axis == 'x':
                    self.move_to_position(0,new_x_pos)
                    self.move_to_position(2,self.main_window.Pos_X1_Scan)
                    self.check_position_and_start(2,self.main_window.Pos_X1_Scan)
                elif self.main_window.axis == 'y':
                    self.move_to_position(2,new_x_pos)
                    self.move_to_position(0,self.main_window.Pos_Y1_Scan)
                    self.check_position_and_start(0,self.main_window.Pos_Y1_Scan)
                else:
                    print("ERROR: invalid axis: {self.main_window.axis}")
        
        # Otherwise, if end of line is not reached yet:
        else:
            #Deviation estimation, vectorized by row by pixel
            factor_PSD = 2 / (self.main_window.number_of_samples * self.main_window.Laser_Frequency)
            self.main_window.Pixel_by_Row = self.main_window.Freq_Data.pow(2).mul(factor_PSD)
            M0_Pixel=self.main_window.Pixel_by_Row.sum(axis=0)
            M0_Pixel = M0_Pixel.replace(0, 1)
            M1_Pixel=self.main_window.Pixel_by_Row.mul(self.main_window.dataFreq, axis=0)
            M1_Pixel=M1_Pixel.sum(axis=0)
            M_dev=M1_Pixel/M0_Pixel
            
            #Set N of data to average fix length
            if self.main_window.Samples_To_AVG_Flag==True:
                self.main_window.Samples_To_AVG=len(M1_Pixel)
                self.main_window.Samples_To_AVG_Flag=False

            M_dev = pd.Series(np.resize(M_dev.to_numpy(), self.main_window.Samples_To_AVG))
            self.main_window.Moment_Dev=pd.concat([self.main_window.Moment_Dev, M_dev], axis=1)
            self.main_window.dataAmp_Avg=(self.main_window.Freq_Data.mean(axis=1))
            
            #Save Avg Spectrum
            self.main_window.Data_Spectrum_Array=pd.concat([self.main_window.Data_Spectrum_Array, self.main_window.dataAmp_Avg], axis=1)
            
            if self.main_window.dataAmp_Avg.empty:
                self.main_window.dataAmp_Avg=self.main_window.dataFreq*0
                self.main_window.Freq_Data=self.main_window.dataFreq*0

            # Make PSD discrete, as original equation is time domain and
            # contains integral to infinity:
            self.main_window.PSD_Avg_Moment=(self.main_window.dataAmp_Avg*self.main_window.dataAmp_Avg)*(2/(self.main_window.number_of_samples*self.main_window.Laser_Frequency))
            
            # Solve M0: simple sum of PSD.
            M0 = np.sum(self.main_window.PSD_Avg_Moment) 
            
            # Small epsilon value to avoid division by near-zero values
            epsilon = 1e-10
            # Solve division by near-zero.
            if np.abs(M0) < epsilon:
                M0 = epsilon  # Set M0 to epsilon instead of 1 to avoid large deviations

    
            # Solve M1: multiplication of the frequency and the PSD.
            M1 = np.sum(self.main_window.dataFreq * self.main_window.PSD_Avg_Moment)
            
            # NOTE: Data debug prints
            print('AVG Moment')
            print(M1/M0)
            print('N Avg Samples')
            print(self.main_window.Counter_DAQ_samples)
            print('M1:')
            print(M1)
            
            # Save calculated moment to variable for flow velocity profile view.
            # TODO: These two are double, and perform the same, but keeping it like this for now to make sure nothing breaks.
            moment = M1/M0
            
            colorcolor=(M1)
            
            # For simulation purposes, artificially increases the moment so the threshold is exceeded.
            # TODO: remove or disable when applying edge scan
            if self.main_window.edge_scan_mode is True:
                self.main_window.simulation_counter += 1
                if self.main_window.simulation_counter == 20:
                    self.main_window.simulation_counter = 0
                    moment = 70000

            # Reset count data average and vector.
            self.main_window.Counter_DAQ_samples=0
            self.main_window.Freq_Data=pd.DataFrame()

            # Determine color given to flow velocity profile pixel.
            # Format is [R,G,B] so the pixel will be varying intensity of green.
            # colorcolor => integer value of average moment.
            self.main_window.New_Color=[0, self.interpolation_color(colorcolor), 0]
            
            # Apply new color to pixel in flow velocity profile view
            # and save measured momentum to CSV file            
            if self.main_window.axis == 'x':
                self.main_window.color_grid_widget.changeCellColor(self.main_window.counter_Data_per_Pixel,self.main_window.counter_Step_Zaber_X, self.main_window.New_Color)
                self.main_window.color_grid_widget.updateCSV(self.main_window.counter_Step_Zaber_X-1,self.main_window.counter_Data_per_Pixel-1,(M1/M0),M0,M1)
            elif self.main_window.axis == 'y':
                self.main_window.color_grid_widget.changeCellColor(self.main_window.counter_Step_Zaber_X,self.main_window.counter_Data_per_Pixel, self.main_window.New_Color)
            else:
                print("ERROR: invalid axis: {self.main_window.axis}")
            
            # 
            self.main_window.counter_Data_per_Pixel+=1
            
            # Change current_column to next pixel (next column)
            self.main_window.current_col= self.main_window.counter_Data_per_Pixel

            # Why increment with one if the max width has been exceeded?
            if (self.main_window.axis == 'x' and self.main_window.counter_Data_per_Pixel >= self.main_window.g_W
                or self.main_window.axis == 'y' and self.main_window.counter_Data_per_Pixel >= self.main_window.g_H):
                self.main_window.counter_Data_per_Pixel += 1
                
            # if edge has been found during edge scan mode
            if self.main_window.edge_scan_mode is True and moment > self.main_window.detect_edge_threshold:    
                    # Stop DAQ
                    if hasattr (self.main_window , 'threadDAQ'):
                        DAQ_Reader_Global.Stop_DAQ_algorithm()
                    
                    # Stop Zabers
                    self.stop_zaber(1)
                    self.stop_zaber(3)
                    
                    # Stop DAQ timer
                    self.main_window.Adquisit_Timer.stop()
                    
                    # Save positions
                    x = self.main_window.scan_functions_instance.get_current_position(0) # Get current position of X Zaber
                    y = self.main_window.scan_functions_instance.get_current_position(2) # Get current position of Y Zaber
                    edge_coordinate = (x,y)
                    
                    # Add coordinate to list of coordinates, currently limited to 4 indexes with each being one direction.
                    self.main_window.edge_coordinate_array[self.main_window.edge_scan_count] = edge_coordinate
   
                    print(f"EDGE SCAN MODE: moment = {moment}")
                    print(f"EDGE SCAN MODE: threshold = {self.main_window.detect_edge_threshold}")
                    print(f"EDGE SCAN MODE: edge found at x = {x} um and y = {y} um")
                    print(f"EDGE SCAN MODE: {self.main_window.edge_coordinate_array[self.main_window.edge_scan_count]} added to coordinate array")

                    # Run edge scan in next direction, or do nothing if all edges have been scanned.
                    self.main_window.edge_scan_count += 1
                    if self.main_window.edge_scan_count <= 3:
                        scan_area_module.scan_all_edges(self.main_window, self.main_window.edge_scan_start_coordinates, self.main_window.edge_scan_count)


    def determine_center_position(self):
        """Starts the edge scan function with a given center position.
        NOTE: this is not fully implemented yet, it finds the edge coordinates (theoretically) but further functions are not implemented.
        """
        self.main_window.edge_scan_start_coordinates = (25000, 25000) # center
        self.main_window.edge_scan_count = 0
        scan_area_module.edge_scan(self.main_window, 'x', self.main_window.edge_scan_start_coordinates)


def Set_Scanning_Tab(self, scan_functionality):
    """Functionality for the scanning tab in the widget
    WARNING: the Scan_functions class is dependent on this function
    TODO: All of this can be moved to the Scan_functions class.
    """
    #Initialization
    # User input settings
    self.Pos_Y1_Scan = float(self.ui.load_pages.lineEdit_y1_scan.text())
    self.Pos_Y2_Scan = float(self.ui.load_pages.lineEdit_y2_scan.text())
    self.speed = float(self.ui.load_pages.lineEdit_speed_ums.text()) * 1000 * 1.6381 / 1.9843
    self.time_timer_scan=0
    self.cell_size = int(self.ui.load_pages.lineEdit_Pixel_size.text())
    # Algorithm Override input settings, will be changed by algorithms
    self.override_user_settings = False 
    # ^if True, the User Input settings from interface will not be used
    self.Pos_X1_Scan_override = float(25000) # center
    self.Pos_X2_Scan_override = float(25000)
    self.Pos_Y1_Scan_override = float(25000)
    self.Pos_Y2_Scan_override = float(25000)
    self.speed_override  = float(200)
    self.time_timer_scan_override =0 # not sure if this override is required
    self.cell_size_override  = int(200)
    
    self.counter_Data_per_Pixel=0 # Used for keeping track of current pixel
    self.current_col=0
    self.current_row=0
    self.New_Color= [255,255,255]
    self.scanning_Finish=False
    #Counter for the N. of average samples: 
    self.Counter_DAQ_samples=0
    self.PSD_Avg_Moment=0

    # Scan Area Module
    self.detect_edge_threshold = 60000 # check if this is correct
    self.edge_scan_mode = False
    self.edge_scan_count = 0
    self.edge_coordinate_array = [tuple()]*4 # create list of tuples
    self.simulation_counter = 0 # delete later
    self.edge_scan_start_coordinates = (25000, 25000)
    self.axis = 'x'

    def Capture_Data_Avg():
        try:
        ######### Update Laser data FFT and Voltage
            self.DAQ_Data=self.threadDAQ.DAQ_Data
            self.data_DAQ2=butter_bandpass_filter(self.DAQ_Data, self.low_freq_filter, self.high_freq_filter,self.order_filter, self.Laser_Frequency)
            self.dataAmp, self.dataFreq, _=FFT_calc(self.data_DAQ2, self.Laser_Frequency)
            amp=pd.Series((self.dataAmp))
            #Add a new column
            self.Freq_Data=pd.concat([self.Freq_Data, amp], axis=1)
            self.Counter_DAQ_samples=self.Counter_DAQ_samples+1
        except:
            print("error in Capture_Data_Avg")

    ###################################################################
    ###################################################################
    #CALIBRATION
    def Start_Vel_Calib():

        #Parameters ###########################
        self.Directory=(self.ui.load_pages.lineEdit_Directory.text())
        self.FileName=(self.ui.load_pages.lineEdit_Name_Files.text())
        self.Amp_Peak_Search=float(self.ui.load_pages.Vel_Threshold_Peak.text())

        zab_start=int(self.ui.load_pages.Vel_Manual_X1.text())
        zab_stop=int(self.ui.load_pages.Vel_Manual_X2.text())
        zab_steps=int(self.ui.load_pages.Vel_Steps_Calib.text())
        self.Data_To_Check=int(self.ui.load_pages.lineEdit_Avg_FFT.text())
        self.Zaber_Steps=np.linspace(start=zab_start, stop=zab_stop, num=zab_steps)
        self.ActualStep=0
        self.Zaber_Pos=0
        self.Vel_Move_Start=False
        self.Freq_Data=pd.DataFrame()
        self.Counter_DAQ_samples=0
        #List with max peaks
        self.Vel_Calib_Peaks_List=[]
        self.Vel_Routine.start()


    def Vel_Routine():
        #Pipeline
        if self.ActualStep<len(self.Zaber_Steps):
            #Move Zaber
            if self.Vel_Move_Start==False:
                with Connection.open_serial_port(self.Zaber_COM) as connection:
                    device_list = connection.detect_devices()
                    new_position=self.Zaber_Steps[self.ActualStep]
                    #zaber_pos=new_position/10000
                    device_list[2].move_absolute(new_position, Units.LENGTH_MICROMETRES)
                    print("---------- Zaber Pos: ---------- " + str(self.Zaber_Steps[self.ActualStep])+" ----------")
                    self.Vel_Move_Start=True
                    #Clear
                    self.Freq_Data=pd.DataFrame()
                    self.Counter_DAQ_samples=0
                    self.Adquisit_Timer.start()
            #Adquisition
            else:
                # Once the samples are acquired
                if self.Counter_DAQ_samples>self.Data_To_Check:
                    self.Adquisit_Timer.stop()
                    self.Vel_Calib_Peaks_List.append(max(self.Freq_Data.mean(axis=1)))
                    #Move to next Position
                    self.ActualStep=self.ActualStep+1
                    print("--- Next Step ---")
                    self.Vel_Move_Start=False
        else:
            self.Vel_Routine.stop()
            print(self.Vel_Calib_Peaks_List)
            #Find Frontiers
            New_Limits=[]
            for i, valor in enumerate(self.Vel_Calib_Peaks_List):
                if valor >= self.Amp_Peak_Search:
                    New_Limits.append(i)
            inicio_grupo = 0
            fin_grupo = 0
            if len(New_Limits) > 2:

                max_longitud = 0
                for i in range(len(New_Limits) - 1):
                    longitud_actual = New_Limits[i+1] - New_Limits[i]
                    if longitud_actual > max_longitud:
                        max_longitud = longitud_actual
                        inicio_grupo = New_Limits[i]
                        fin_grupo = New_Limits[i+1]
            New_Limits=[inicio_grupo, fin_grupo]
            print(New_Limits)
            self.ui.load_pages.Vel_Limit_X2.setText(str(round(self.Zaber_Steps[fin_grupo],1)))
            self.ui.load_pages.Vel_Limit_X1.setText(str(round(self.Zaber_Steps[inicio_grupo],1)))

            print("---------- Calibration finished ---------- ")



    def Get_Report():
        plot_grouped_error_bars('.\Temporal_Files\Scanning\Scanning_Moments_Dev.csv', int((self.Pos_Y2_Scan-self.Pos_Y1_Scan)/self.cell_size))

    ###################################################################
    ###################################################################
    self.Data_Spectrum_Array=pd.DataFrame()
    
    # Timer connections
    self.Pixel_Interval = QTimer()
    self.Pixel_Interval.timeout.connect(scan_functionality.change_pixel_daq)

    # Continuous Sweep Routine
    # Configure timer that determines sampling frequency
    # NOTE: This timer determines how often the program requests the sampling data
    # from the DAQ. After every interval, all the samples of the past interval are
    # requested, and then the average is taken from that.
    # NOTE: interval used to be 1ms, but had to be changed to 10 due to laptop
    # performance. With a faster laptop, it can be returned to 1.
    # NOTE: to compensate for this interval change, reduce the measurement speed
    # by the same factor (so, 1/10th for example) if you want to keep the same samples / pixel.
    # At the moment it has around 300 samples per pixel of 200 um, so this 
    # can be reduced without issue.
    self.Adquisit_Timer = QTimer()
    self.Adquisit_Timer.setInterval(10) # ISSUE here, works with 5 but unstable.
    self.Adquisit_Timer.timeout.connect(Capture_Data_Avg)

    #For Calibration Routine
    # Configure timer that determines sampling frequency
    self.Vel_Routine= QTimer()
    self.Vel_Routine.setInterval(1)
    self.Vel_Routine.timeout.connect(Vel_Routine)

    # Link buttons to functions #
    self.ui.load_pages.Stop_x_but.clicked.connect(scan_functionality.get_current_position)
    self.ui.load_pages.Vel_Start_Calib.clicked.connect(Start_Vel_Calib)
    self.ui.load_pages.Vel_Report.clicked.connect(Get_Report)
    ###################################################################
    ###################################################################

