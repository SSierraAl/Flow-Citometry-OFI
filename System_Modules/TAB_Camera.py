
###########################################################################################################################
################################################# Libraries ###############################################################
###########################################################################################################################
from PyQt6.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from System_Modules.Camera_Selector_Fn import camera_selector
from pyphantom import Phantom, utils, cine
from collections import deque
import time
import numpy as np
import os
import cv2
import keyboard
###########################################################################################################################


class WorkerCamera(QObject):
    # Signal to emit captured frames
    #frames_captured = Signal(np.ndarray)
    capture_finished = Signal(list)

    def __init__(self, cam,resolution,trigger_frames,snapshots_folder,image_label,global_counter_cam):
        super().__init__()
        self.cam = cam
        self.global_counter_cam=global_counter_cam
        self.resolution =resolution
        self.snapshots_folder=snapshots_folder
        self.image_label=image_label
        self.trigger_frames=trigger_frames
        self.cam_running = False
        self.frames = []
        self.particle_signal = False
        self.particle_signal_prev = False
        self.frame_count = 0
        self.h=self.resolution[1]
        self.w=self.resolution[0]
        self.ch=3
        self.bytes_per_line = self.ch * self.w

    @Slot()
    def capture_frame(self):
        while self.cam_running == True:
            live_image = self.cam.get_live_image()
            cv2.waitKey(20)  
            live_image = (live_image / 256).astype(np.uint8)
            qt_img = self.convert_cv_qt(live_image)
            self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, rgb_image):
        """Convert image to QPixmap"""
        convert_to_Qt_format = QImage(rgb_image.data, self.w, self.h, self.bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.resolution[0], self.resolution[1], Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)



    #camera record for the button additional one:
    def signal_Record(self):
        self.cam.record()                                    
    #camera trigger for the button additional two:

    def signal_Trigger(self):
        self.cam.trigger()

    def stop(self):
        self.cam_running = False
        #cv2.destroyAllWindows

    def startstart(self):
        self.cam_running=True


############################################################################################################################
######################################## Class for Managing Frame Capture ##################################################
############################################################################################################################

class FrameCapture(QObject):
    workCAM_requested = Signal(int)
    def __init__(self,main_window):
        super().__init__()

        self.main_window=main_window
        self.global_counter_cam=0
        self.ph = Phantom() 
        #Interfaz buttons connections #######################################
        #Start all the process
        self.main_window.ui.load_pages.but_start_cam_2.clicked.connect(self.but_start_cam)
        #Stop all the process and kill the thread
        self.main_window.ui.load_pages.but_stop_cam_2.clicked.connect(self.close_capture_cam)
        #start recording(in other words activating the capture button in the pcc software)
        self.main_window.ui.load_pages.but_record_cam.clicked.connect(self.record_cam)
        #start triggering(in other words activating the trigger button in the pcc software)
        self.main_window.ui.load_pages.but_trigger_cam.clicked.connect(self.trigger_cam)


    #Start all the process, and read the parameters, be carefull after we need to be able to stop everything
    def but_start_cam(self):

        self.cam_count = self.ph.camera_count                     # Check for connected cameras
        self.cam = camera_selector(self.ph)                       # Connect to the specific camera
        self.cam = self.ph.Camera(0)
        # Now we have at least one camera connected   
        self.ph.discover(print_list=True)
        print('------ Camera Reading Mode: ON -------')

        #Update parameters based on the GUI
        #self.cam.partition_count  = int(self.main_window.ui.load_pages.line_partition_count_cam_2.text())
        #self.cam.frame_rate = int(float(self.main_window.ui.load_pages.line_frame_rate_cam_2.text()))
        #Number of images to save
        #self.trigger_frames = int(self.main_window.ui.load_pages.line_trigger_frame_cam_2.text())
        #self.snapshots_folder = self.main_window.ui.load_pages.line_directory_cam_2.text()
        #self.cam.resolution = (int(self.main_window.ui.load_pages.line_resolution_cam_w_2.text()),int(self.main_window.ui.load_pages.line_resolution_cam_h_2.text()))
        #self.cam.exposure =int(float(self.main_window.ui.load_pages.line_exposure_cam_2.text()))
        #Resize display widget
        #self.main_window.ui.load_pages.image_label.resize(self.cam.resolution[0],self.cam.resolution[1])


        # Check parameters ###################################################################
        res = self.cam.resolution                                 
        self.main_window.ui.load_pages.line_resolution_cam_w_2.setText(str(res[0])) 
        self.main_window.ui.load_pages.line_resolution_cam_h_2.setText(str(res[1])) 
        frame_rate = self.cam.frame_rate   
        self.main_window.ui.load_pages.line_frame_rate_cam_2.setText(str(frame_rate))                       
        exposure = self.cam.exposure    
        self.main_window.ui.load_pages.line_exposure_cam_2.setText(str(exposure))
        partition_count = self.cam.partition_count
        self.main_window.ui.load_pages.line_partition_count_cam_2.setText(str(partition_count))                       
        #####################################################################################

        #Initialize Class
        """
        self.frames = []
        self.worker_Camera = WorkerCamera(self.cam,self.cam.resolution,self.trigger_frames,self.snapshots_folder,self.main_window.ui.load_pages.image_label,self.global_counter_cam)
        self.workCAM_requested.connect(self.worker_Camera.capture_frame)
        self.thread_camera = QThread()
        self.worker_Camera.moveToThread(self.thread_camera)
        #Start thread (init)
        self.thread_camera.start()
        #Set cam_running variable in True
        self.worker_Camera.startstart()
        # Call the infinit loop to request the camera
        self.workCAM_requested.emit(0)
        """



    #Stop timmer and terminate the thread whtn the app is closed
    def close_capture_cam(self):
        try:
            self.cam.close()
            #self.ph.close()  
            self.worker_Camera.stop()        
            self.thread_camera.quit()
            self.thread_camera.wait()
            print('------ Camera Reading Mode: Off -------')
        except:
            pass

    def trigger_particle_signal(self):
        self.worker_Camera.signal_P()

    def trigger_particle_signal_prev(self):
        
        self.cam.is_busy()
        #self.worker_Camera.signal_P_Prev()

    def record_cam(self):
        self.cam.record()
        #self.worker_Camera.signal_Record()

    def trigger_cam(self):

        self.cam.trigger()
        #self.worker_Camera.signal_Trigger()



