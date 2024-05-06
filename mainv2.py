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

#import keyboard

def is_near_traffic_light(x_gps, y_gps, traffic_light_area, threshold=0.05):
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Revisar cada semáforo en la lista
    for traffic_light in traffic_light_area:
        x_light, y_light = traffic_light
        if distance(x_gps, y_gps, x_light, y_light) <= threshold:
            return True     
    return False  

def segmentar_pista(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización para separar las partes claras de las oscuras
    _, umbralizada = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)
    
    # Aplicar un filtro de mediana para eliminar el ruido
    filtrada = cv2.medianBlur(umbralizada, 5)
    
    # Definir una máscara para enfocarnos en la pista (líneas negras y amarillas)
    mascara = np.zeros_like(filtrada)
    blanco = (255,)
    cv2.inRange(filtrada, blanco, blanco, mascara)
    
    # Aplicar la máscara
    pista = cv2.bitwise_and(filtrada, filtrada, mask=mascara)
    
    # Detectar bordes usando el algoritmo Canny
    bordes = cv2.Canny(pista, 50, 150)
    return bordes

def define_region(image):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    # Vertices array of the unwanted area (bottom 1/3 of the picture)
    region = [np.array([(0, height), (0, height//3), (width, height//3), (width, height)])]

    return region

def crop_frame(image):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    return image[0: height//3*2, 0: width]

def mask_image(image, vertices):
    # Fill unwanted area with zeros
    cv2.fillPoly(image, vertices, 0)
    return image

def detect_traffic_light(image,pos):
    detection=False
    color = 'r'

    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape

    # Definir la mitad superior de la imagen como región de interés (ROI)
    roi = image[0:height//2, :]

    # Convertir la ROI de BGR a HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Definir el rango de color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Crear una máscara que solo contenga tonos amarillos en la ROI
    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # Encontrar contornos amarillos en la máscara
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours_yellow)>0):
        if -2.164<pos[0]<-1.74 and 1.633<pos[1]<2.161 or 1.998<pos[0]<2.363 and -0.325<pos[1]<0.242:
            detection=True
        # Dibujar rectángulos alrededor de los contornos amarillos encontrados
        for cnt in contours_yellow:
            x, y, w, h = cv2.boundingRect(cnt)
            # Ajustar las coordenadas del rectángulo para la imagen completa
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Definir la ROI dentro del rectángulo amarillo
            roi_red = image[y:y+h, x:x+w]
            # Convertir la ROI de BGR a HSV
            hsv_roi_red = cv2.cvtColor(roi_red, cv2.COLOR_BGR2HSV)
            # Definir el rango de color rojo brillante en HSV
            lower_red = np.array([0, 220, 220])
            upper_red = np.array([10, 255, 255])
            
            # Crear una máscara que solo contenga tonos rojos brillantes en la ROI
            mask_red = cv2.inRange(hsv_roi_red, lower_red, upper_red)
            
            # Encontrar contornos rojos brillantes en la máscaraws
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Si se detecta rojo, poner texto "ROJO" en el rectángulo amarillo, sino "VERDE"
            if len(contours_red) > 0:
                color='r'
                cv2.putText(image, "ROJO", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                color='g'
                cv2.putText(image, "VERDE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image,detection,color




#traffic light position
traffic_light1=[2.223,-0.24]
traffic_light2=[-1.94,2.11]
traffic_light_area=[traffic_light1,traffic_light2]

# Initialize RGB camera 
myCam = QCarRealSense(mode='RGB, Depth')

# Initialize car
myCar=QCar(frequency=200) 

x0 = np.zeros((3,1))
gps = QCarGPS(initialPose=x0)

#Carga de modelo
model=load_model('Modelo0605.h5')

#Niveles de giro
data_cat=['-0.1', '-0.3', '0.0', '0.1', '0.3']

#stop_sign_cascade = cv2.CascadeClassifier("stop_sign_classifier_2.xml")


while True:
    #Set speed and turn
    speed=0.1
    turn=0.1
    traffic_light_zone=False

    #Read GPS
    if(gps.readGPS()):
        x_gps = gps.position[0]
        y_gps = gps.position[1]
        #print(x_gps,y_gps)
        traffic_light_zone=is_near_traffic_light(x_gps, y_gps, traffic_light_area, threshold=0.25)



    #Read camera
    myCam.read_RGB()

    #Save images
    imagen=myCam.imageBufferRGB
    imagen = cv2.resize(imagen, (320, 240))
    imagen=imagen[120:240,0:320]
    imagen= segmentar_pista(imagen)
    imagen = np.expand_dims(imagen, axis=-1)
    imagen = np.expand_dims(imagen, axis=0)
    imagen = np.concatenate([imagen] * 3, axis=-1)
    
    prediction=model.predict(imagen)
    score = tf.nn.softmax(prediction)
    new_turn=data_cat[np.argmax(score)]
    new_turn=float(new_turn)

    myCar.write(speed, new_turn)
    
    
    
    #Show images
    #cv2.imshow('My RGB', image)
    #cv2.waitKey(1)


    

    



