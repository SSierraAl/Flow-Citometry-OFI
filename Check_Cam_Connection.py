from Libraries import *

class Phantom_Cam():
    'Class to manage the connection with the camera and its parameters'
    def __init__(self):
        pass

        self.ph = Phantom()                                       #make a phantom object
        self.cam = self.camera_selector(self.ph)                       #to connect to specific camera
        print("--- Connected to the existing camera ---")
        self.cam = self.ph.Camera(0)                        
        self.ph.discover(print_list = True)  #[name, serial number, model, camera number]
        
    

    #Camera selecction function
    def camera_selector(self, ph):   #ph is a Phantom() object
        cam_count = ph.camera_count
        if cam_count == 0:                    #case no camera is connected, makes simulated camera
            print("No cameras connected, making simulated camera")
            ph.add_simulated_camera()
            cam = ph.Camera(0)
        elif cam_count == 1:                  #case exactly 1 camera is connected, just connects
            print("Connected to only connected camera")
            cam = ph.Camera(0)
        else:                                 #case more than 1 camera is connected, walks through the connected cameras and prompts you to choose the one to connect to                                      
            for i in range(cam_count):
                cam_model = ph.discover()[i][2]    #ph.discover()[i][name, serial number, model, camera number]
                cam_sn = ph.discover()[i][1]
                use_cam = input("Camera %s is a %s with serial number %s, is this the camera you would like to use? y or n? \n" % (i, cam_model, cam_sn))
                try: 
                    if use_cam == "y":
                        cam = ph.Camera(i)
                        break
                except: 
                    print("That's not a legal input")
        try: 
            cam.get_partition_state(0)          #just testing something innocuous to see if camera selected, if something more innocuous can be used here, should do that
        except:
            print("No camera selected")
        return cam



    def get_camera_params(self):
        # #Get parameters
        res=self.cam.resolution                                 #geting resolution
        print ('{:d} {} {:d} {}'.format(res[0],'x',res[1],'Resolution'))
        frame_rate = self.cam.frame_rate                         #geting frame rate
        print("%s Framerate(fps)" % (frame_rate))
        part_count = self.cam.partition_count                    #get partition count 
        print("%s Partition_count" % (part_count))
        exposure = self.cam.exposure                             #get exposure time
        print("%s exposure(us)" % (exposure))


    def trigger_cam(self):
        #Trigger the camera
        self.cam.trigger()    