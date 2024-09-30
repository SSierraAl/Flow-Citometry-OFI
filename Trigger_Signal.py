#Code to test the external trigger function of the Phantom Camera using the DAQ NI USB-6361

from Libraries import *

# Define the DAQ device and channel
device_name = 'Dev1'  # Replace with your DAQ device name
output_channel = f'{device_name}/ao0'  # Analog output channel 0

# Create a Task to handle the analog output
with nidaqmx.Task() as task:
    # Add an analog output voltage channel
    task.ao_channels.add_ao_voltage_chan(output_channel, min_val=0.0, max_val=5.0)

    # Function to set the voltage
    def set_voltage(voltage):
        task.write(voltage)
        print(f"Voltage set to {voltage}V")

    # Initialize the last voltage state
    last_voltage = 5.0
    set_voltage(last_voltage)

    print("Press 'a' to toggle voltage (0V when pressed, 5V when released)")


    #########################################################################
    # Periodic Signal to Test
    #########################################################################
    while True:

        task.write(5)
        task.write(0)

    ##########################################################################
    # Toggle 'a' press and release
    ########################################################################## 
    try:
        while True:
            # Detect spacebar press and release
            if keyboard.is_pressed('a') and last_voltage != 0.0:
                # Set to 0V if spacebar is pressed and the last voltage wasn't 0V
                set_voltage(0.0)
                last_voltage = 0.0
            elif not keyboard.is_pressed('a') and last_voltage != 5.0:
                # Set to 5V if spacebar is released and the last voltage wasn't 5V
                set_voltage(5.0)
                last_voltage = 5.0

    except KeyboardInterrupt:
        print("Program terminated.")
