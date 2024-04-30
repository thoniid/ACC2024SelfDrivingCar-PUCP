import os
import numpy as np
import time
import cv2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.flooring import QLabsFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
import pal.resources.rtmodels as rtmodels
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from PIL import Image

# Try to connect to Qlabs
os.system('cls')
qlabs = QuanserInteractiveLabs()
print("Connecting to QLabs...")
try:
    qlabs.open("localhost")
    print("Connected to QLabs")
except:
    print("Unable to connect to QLabs")
    quit()

# Delete any previous QCar instances and stop any running spawn models
qlabs.destroy_all_spawned_actors()
# QLabsRealTime().terminate_all_real_time_models()

#Set the Workspace Title
hSystem = QLabsSystem(qlabs)
x = hSystem.set_title_string('ACC Self Driving Car Competition', waitForConfirmation=True)


### Flooring

x_offset = 0.13
y_offset = 1.67
hFloor = QLabsFlooring(qlabs)
#hFloor.spawn([0.199, -0.491, 0.005])
hFloor.spawn_degrees([x_offset, y_offset, 0.001],rotation = [0, 0, -90])


### region: Walls
hWall = QLabsWalls(qlabs)
hWall.set_enable_dynamics(False)

for y in range (5):
    hWall.spawn_degrees(location=[-2.4 + x_offset, (-y*1.0)+2.55 + y_offset, 0.001], rotation=[0, 0, 0])

for x in range (5):
    hWall.spawn_degrees(location=[-1.9+x + x_offset, 3.05+ y_offset, 0.001], rotation=[0, 0, 90])

for y in range (6):
    hWall.spawn_degrees(location=[2.4+ x_offset, (-y*1.0)+2.55 + y_offset, 0.001], rotation=[0, 0, 0])

for x in range (5):
    hWall.spawn_degrees(location=[-1.9+x+ x_offset, -3.05+ y_offset, 0.001], rotation=[0, 0, 90])

hWall.spawn_degrees(location=[-2.03 + x_offset, -2.275+ y_offset, 0.001], rotation=[0, 0, 48])
hWall.spawn_degrees(location=[-1.575+ x_offset, -2.7+ y_offset, 0.001], rotation=[0, 0, 48])


# Spawn a QCar at the given initial pose
car2 = QLabsQCar(qlabs)
car2.spawn_id_degrees(actorNumber=0, location=[-1.335+ x_offset, -2.5+ y_offset, 0.005], rotation=[0, 0, -45], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
basicshape2 = QLabsBasicShape(qlabs)
basicshape2.spawn_id_and_parent_with_relative_transform(actorNumber=102, location=[1.15, 0, 1.8], rotation=[0, 0, 0], scale=[.65, .65, .1], configuration=basicshape2.SHAPE_SPHERE, parentClassID=car2.ID_QCAR, parentActorNumber=2, parentComponent=1,  waitForConfirmation=True)
basicshape2.set_material_properties(color=[0.4,0,0], roughness=0.4, metallic=True, waitForConfirmation=True)

camera1=QLabsFreeCamera(qlabs)
camera1.spawn_degrees (location = [-0.426+ x_offset, -5.601+ y_offset, 4.823], rotation=[0, 41, 90])

camera2=QLabsFreeCamera(qlabs)
camera2.spawn_degrees (location = [-0.4+ x_offset, -4.562+ y_offset, 3.938], rotation=[0, 47, 90])

camera3=QLabsFreeCamera(qlabs)
camera3.spawn_degrees (location = [-0.36+ x_offset, -3.691+ y_offset, 2.652], rotation=[0, 47, 90])

camera2.possess()

# stop signs
myStopSign = QLabsStopSign(qlabs)
myStopSign.spawn_degrees ([2.25 + x_offset, 1.5 + y_offset, 0.05], [0, 0, -90], [0.1, 0.1, 0.1], False)
myStopSign.spawn_degrees ([-1.3 + x_offset, 2.9 + y_offset, 0.05], [0, 0, -15], [0.1, 0.1, 0.1], False)

# Spawning crosswalks
myCrossWalk = QLabsCrosswalk(qlabs)
myCrossWalk.spawn_degrees (location =[-2 + x_offset, -1.475 + y_offset, 0.01],
            rotation=[0,0,0], scale = [0.1,0.1,0.075],
            configuration = 0)

mySpline = QLabsBasicShape(qlabs)
mySpline.spawn_degrees ([2.05 + x_offset, -1.5 + y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)
mySpline.spawn_degrees ([-2.075 + x_offset, y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)

TrafficLight0 = QLabsTrafficLight(qlabs)
TrafficLight0.spawn_degrees([2.3 + x_offset, y_offset, 0], [0, 0, 0], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
TrafficLight1 = QLabsTrafficLight(qlabs)
TrafficLight1.spawn_degrees([-2.3 + x_offset, -1 + y_offset, 0], [0, 0, 180], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)

# Start spawn model
QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

#Funciones
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

#Carga de modelo
model=load_model('ModelFuncional3.h5')

#Variables de velocidad y giro 
forward = 0
turn = 0
speed=0.5

#Niveles de giro
data_cat=['-0.1', '-0.2', '0.0', '0.1', '0.2']

#Flags counters
count = 0
i=0
stop=1
initial_time=0
detection=False
stop_sign_cascade = cv2.CascadeClassifier("stop_sign_classifier_2.xml")
pos=[0,0,0]
#Logica de conduccion autonoma
while True:

    #Alternancia de luces de semaforo
    if count >= 50:
        count = 0
    if count<=10:
        TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
        TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
    else:
        TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
        TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
    count = count + 1

    #Carga y preprocesamiento de la imagen
    x, camera_image1 = car2.get_image(camera=car2.CAMERA_RGB)
    #imagen = cv2.resize(camera_image1, (160, 120))
    #imagen=imagen[120:240,0:320]
    imagen= segmentar_pista(camera_image1)
    imagen = np.expand_dims(imagen, axis=-1)
    imagen = np.expand_dims(imagen, axis=0)
    imagen = np.concatenate([imagen] * 3, axis=-1)
    
    #Cargar modelo
    prediction=model.predict(imagen)
    #print(prediction)
    score = tf.nn.softmax(prediction)
    new_turn=data_cat[np.argmax(score)]

    if speed==0 and stop==0:
            time.sleep(5)
            #Tiempo a detenerse por señal de stop
            speed=0.5
    
    #Mostrar en consola velocidad y nivel de giro
    print("Angulo giro:",new_turn)
    print("Velocidad:",speed)
    #Logica de deteccion de señal stop
    if x:
        #Preprocesamiento para deteccion de stop
        cropped_frame = crop_frame(camera_image1)    
        img_filter = cv2.GaussianBlur(cropped_frame, (5, 5), 0)
        gray_filtered = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

        #Objeto con ubicacion de la señal detectada en la imagen
        stop_signs = stop_sign_cascade.detectMultiScale(gray_filtered, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))

        final_time=time.time()
        
        #Condicion para detenerse por señal de stop
        if stop and stop_signs != ():
            print("Señal de stop detectada")
            speed=0
            initial_time=time.time()
            stop=0
        
        #Tiempo necesario para volver a detectar una nueva señal y no detecte la misma varias veces
        if final_time-initial_time>8:
            stop=1
    
    #Logica deteccion semaforo
    image,detection,color=detect_traffic_light(camera_image1,pos)
    print("Color de semaforo:",color)
    if detection and color=='r':
        print("Parada por semaforo rojo")
        speed=0
    elif stop!=0:
        #Ajustar velocidad distinta para tramos rectos y curvos
        if new_turn=='0.0':
            speed=0.5
        else:
            speed=0.3

    #Asignacion de parametros de velocidad y giro al auto
    a,pos,c,d,e=car2.set_velocity_and_request_state(
        forward=speed,
        turn=float(new_turn),
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=True,
        brakeSignal=False,
        reverseSignal=False
    )

    i+=1