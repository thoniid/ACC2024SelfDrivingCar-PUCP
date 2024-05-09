import time
import math
import cv2
from pal.products.qcar import QCar,QCarRealSense,QCarGPS
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import time
from PIL import Image
from hal.products.mats import SDCSRoadMap

#Hyperparameters
SUPER_HIGH_SPEED=0.068
HIGH_SPEED=0.063
LOW_SPEED=0.052
START_SPEED=0.053

#--------Function declaration---------

"""
    Function to calculate the Euclidean distance between two points.

    Parameters:
    - x1: The x-coordinate of the first point.
    - y1: The y-coordinate of the first point.
    - x2: The x-coordinate of the second point.
    - y2: The y-coordinate of the second point.

    Returns:
    - dist: The Euclidean distance between the two points.
"""
def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

"""
    Function to check if a GPS location is within a parking area.

    Parameters:
    - x_gps: The x-coordinate of the GPS location of the QCar.
    - y_gps: The y-coordinate of the GPS location of the QCar.
    - parking_area: A list containing the coordinates of the parking area.

    Returns:
    - in_parking_area: A boolean indicating whether the GPS location is within the parking area.
"""
def parking_stop(x_gps, y_gps,parking_area):
    threshold = 0.08
    x_park,y_park = parking_area
    if distance(x_gps,y_gps,x_park,y_park) <= threshold:
        print("Parking area")
        return True
    return False

"""
    Function to check if a GPS location is near a traffic light.

    Parameters:
    - x_gps: x-coordinate of the GPS location of the QCar.
    - y_gps: y-coordinate of the GPS location of the QCar.
    - traffic_light_area: A tuple containing the coordinates of the traffic light stop area.
    - threshold: The distance threshold within which the GPS location is considered near the traffic light area.

    Returns:
    - near_traffic_light: A boolean indicating whether the GPS location is near a traffic light.
"""
def is_near_traffic_light(x_gps, y_gps, traffic_light_area, threshold=0.05):
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x0,y0=traffic_light_area[0]
    x1,y1=traffic_light_area[1]
    distance_light0=distance(x_gps, y_gps, x0, y0)
    distance_light1=distance(x_gps, y_gps, x1, y1)
    # Review each traffic light in the list
    if distance_light0 <= threshold or distance_light1 <=threshold:
        return True
    return False

"""
    Function to segment the track from an image.

    Parameters:
    - image: Input image in BGR color space.

    Returns:
    - edges: Segmented image with track edges detected.
"""
def segment_track(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to separate light parts from dark parts
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Apply a median filter to remove noise
    filtered = cv2.medianBlur(thresholded, 5)
    
    # Define a mask to focus on the track (black and yellow lines)
    mask = np.zeros_like(filtered)
    white = (255,)
    cv2.inRange(filtered, white, white, mask)
    
    track = cv2.bitwise_and(filtered, filtered, mask=mask)
    
    # Detect edges using the Canny algorithm
    edges = cv2.Canny(track, 50, 150)
    return edges

"""
    Define a region of interest (ROI) to exclude the bottom one-third of an image for the stop signs detection.

    Parameters:
    - image: Input image from which the region of interest (ROI) will be defined.

    Returns:
    - region: List containing vertices of the unwanted area (bottom one-third of the image).
"""
def define_region(image):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    # Vertices array of the unwanted area (bottom 1/3 of the picture)
    region = [np.array([(0, height), (0, height//3), (width, height//3), (width, height)])]

    return region

"""
    Function to crop a frame from the two-thirds of an image for the stop signs detection.

    Parameters:
    - image: Input image to be cropped.

    Returns:
    - cropped_image: Cropped image containing the two-thirds of the input image.
"""
def crop_frame(image):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    return image[0: height//3*2, 0: width]

"""
    Mask an image using specified vertices for the stop signs detection.

    Parameters:
    - image: Input image to be masked.
    - vertices: List containing vertices of the unwanted area.

    Returns:
    - masked_image: Image with the unwanted area filled with zeros.
"""
def mask_image(image, vertices):
    cv2.fillPoly(image, vertices, 0)
    return image

"""
    Function to detect traffic lights in an image and determine their color.

    Parameters:
    - image: Input image containing traffic lights.
    - position: Null array.

    Returns:
    - image: Image with detected traffic lights and their labels.
    - detection: Boolean indicating whether a traffic light is detected in the image.
    - color: Color of the detected traffic light ('r' for red, 'g' for green).
"""
def detect_traffic_light(image, position):
    detection = False
    color = 'r'

    height, width, _ = image.shape

    # Define the upper half of the image as the Region of Interest (ROI)
    roi = image[0:height//2, :]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours_yellow) > 0):
        if -2.164 < position[0] < -1.74 and 1.633 < position[1] < 2.161 or 1.998 < position[0] < 2.363 and -0.325 < position[1] < 0.242:
            detection = True

        # Draw rectangles around the detected yellow contours
        for cnt in contours_yellow:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Define the ROI within the yellow rectangle
            roi_red = image[y:y+h, x:x+w]
            
            hsv_roi_red = cv2.cvtColor(roi_red, cv2.COLOR_BGR2HSV)
            # Define the range of bright red color in HSV
            lower_red = np.array([0, 220, 220])
            upper_red = np.array([10, 255, 255])
            
            mask_red = cv2.inRange(hsv_roi_red, lower_red, upper_red)
            
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_red) > 0:
                color = 'r'
                cv2.putText(image, "RED", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                color = 'g'
                cv2.putText(image, "GREEN", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detection, color

"""
    Function to find the closest waypoint to the current position and determine its speed.

    Parameters:
    - current_x: The current x-coordinate of the vehicle.
    - current_y: The current y-coordinate of the vehicle.
    - waypoints: A list of waypoints, each containing coordinates (x, y).
    - fast_wp: A list of indices indicating waypoints with high-speed limits.
    - slow_wp: A list of indices indicating waypoints with low-speed limits.

    Returns:
    - speed: The speed limit at the closest waypoint (HIGH_SPEED, LOW_SPEED, or None).
"""
def find_closest_waypoint(current_x, current_y, waypoints, fast_wp, slow_wp,super_fast_wp):
    min_distance = float('inf')
    closest_waypoint_index = None
    i = 0
    for waypoint in waypoints:
        x_waypoint, y_waypoint = waypoint
        current_distance = distance(current_x, current_y, x_waypoint, y_waypoint)
        if current_distance < min_distance:
            min_distance = current_distance
            closest_waypoint_index = i
        i += 1
    
    if min_distance <= 0.5:
        if closest_waypoint_index in fast_wp:
            return HIGH_SPEED
        elif closest_waypoint_index in slow_wp:
            return LOW_SPEED
        elif closest_waypoint_index in super_fast_wp:
            return SUPER_HIGH_SPEED
    else:
        return None
    

#-------------- M A I N - P R O G R A M ---------------


# --------Initialization of peripherals----------

# Initialize RGB camera 
myCam = QCarRealSense(mode='RGB, Depth')

# Initialize QCar
myCar=QCar(frequency=200) 

# Initialize GPS
x0 = np.zeros((3,1))
gps = QCarGPS(initialPose=x0)


# --------Machine learning models----------

#Load autonomous navigation model
model=load_model('Modelo0605.h5')

#Turning steering levels
data_cat=['-0.1', '-0.3', '0.0', '0.1', '0.3']

#Load stop sign model
stop_sign_cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')


# --------Coordinates of key positions----------

#Traffic lights
traffic_light1=[2.223,-0.24]
traffic_light2=[-1.94,2.11]
traffic_light_area=[traffic_light1,traffic_light2]

#Parking area
parking_area = [-1.96,0.67]

#Waypoints
waypoints = [
    (-1.205, -0.83),
    (-0.45321888, -1.0708122),
    (1.36, -1.060911),
    (2.1984928, 0.35228203),
    (2.2143471, 3.4026504),
    (0.70, 4.426217),
    (-0.89103985, 4.4064493),
    (-1.9282475, 3.25)
]


# --------Sentinel values----------

red=0
stop=True
parking_zone=False


# --------Control values----------
initial_time=0
speed=START_SPEED
super_fast_wp=[3]
fast_wp=[1,5,7]
slow_wp=[2,4,6]
pos=[0,0,0]
traffic_light_zone=False
parking_zone=False

# --------M A I N - L O O P----------

while True:

    #Get GPS position
    if(gps.readGPS()):
        x_gps = gps.position[0]
        y_gps = gps.position[1]
        #print("Position: ", x_gps,y_gps)
        parking_zone = parking_stop(x_gps, y_gps,parking_area)
        traffic_light_zone=is_near_traffic_light(x_gps, y_gps, traffic_light_area, threshold=0.25)
        
    
    #Read and save images from camera
    myCam.read_RGB()
    image=myCam.imageBufferRGB

    #---------Driving-----------
   
    #Process camera images
    original_image=image.copy()
    image = cv2.resize(image, (320, 240))
    image=image[120:240,0:320]
    image= segment_track(image)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = np.concatenate([image] * 3, axis=-1)
 
    #Turn angle prediction
    prediction=model.predict(image)
    score = tf.nn.softmax(prediction)
    new_turn=data_cat[np.argmax(score)]
    new_turn=float(new_turn)

    #---------STOP signal-----------

    if speed==0 and stop==0:
        #Time to stop due to stop signal
        time.sleep(3)
        speed=HIGH_SPEED
    final_time=time.time()

    #Image preprocessing for stop signal detection
    cropped_frame = crop_frame(original_image)    
    img_filter = cv2.GaussianBlur(cropped_frame, (5, 5), 0)
    gray_filtered = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

    #Stop signal detection
    stop_signs = stop_sign_cascade.detectMultiScale(gray_filtered, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))

    #Stop vehicle if signal is detected
    try:
        if stop and stop_signs != ():
        #print("Stop signal detected")
            speed=0
            initial_time=time.time()
            stop=0
    except:
        pass 
    #Stop signal detection timer
    if final_time-initial_time>6:
        stop=1

    #---------Traffic lights-----------

    image,detection,color=detect_traffic_light(original_image,pos)
    closest_waypoint = find_closest_waypoint(x_gps, y_gps, waypoints,fast_wp,slow_wp,super_fast_wp)
    
    if traffic_light_zone and color=='r':
        red=1
    elif stop!=0 and closest_waypoint:
        if closest_waypoint:
            speed=closest_waypoint

    if red and color == 'r':
        speed=0

    elif red and color=='g':
        speed=HIGH_SPEED
        red=0
    #----------End of the road------
    if parking_zone:
        myCar.write(throttle=0, steering=0)
        break
     
    #-----------Set throttle and steering----------
    myCar.write(throttle=speed, steering=-new_turn)
    
    



    

    



