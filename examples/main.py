import time
import cv2
from pal.products.qcar import QCar,QCarCameras,QCarRealSense
import numpy as np
import time

# Initialize cameras
cameras = QCarCameras(
    enableBack=True,
    enableFront=True,
    enableLeft=True,
    enableRight=True,
)

# Initialize RGB camera 
myCam = QCarRealSense(mode='RGB, Depth')

# Initialize car
myCar=QCar(frequency=200) 

while True:
    #Set speed and turn
    speed=3
    turn=0.5

    #Read cameras
    cameras.readAll()
    csi_images = {}

    myCam.read_RGB()

    #Save images
    image=myCam.imageBufferRGB

    
    #Delete if we don't use CSI cameras
    for i, c in enumerate(cameras.csi):
        if c is not None:
            csi_images[f"CSI_{i}"] = c.imageData.copy()
    
    #Show images
    cv2.imshow('My RGB', image)
    cv2.imshow('CSI', csi_images["CSI_0"])


    #Gooo
    myCar.write(speed, turn)

    cv2.waitKey(100)



