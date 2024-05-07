# region: package imports
import os
import cv2
import keyboard
import numpy as np
# environment objects

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
import pal.resources.rtmodels as rtmodels

i=0

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
#car2.set_transform_and_request_state(headlights=False)
car2.spawn_id_degrees(actorNumber=0, location=[-1.335+ x_offset, -2.5+ y_offset, 0.005], rotation=[0, 0, 135], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
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

# Start spawn model
QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

forward = 0
turn = 0
image_capture=0

def segmentar_pista(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización para separar las partes claras de las oscuras
    _, umbralizada = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)

    # Aplicar un filtro de mediana para eliminar el ruido
    filtrada = cv2.medianBlur(umbralizada, 5)
    
    # Crear una máscara para enfocarnos en los objetos negros
    mascara = np.zeros_like(filtrada)
    blanco = (255,)
    cv2.inRange(filtrada, blanco, blanco, mascara)
    
    # Aplicar la máscara
    pista = cv2.bitwise_and(filtrada, filtrada, mask=mascara)
    
    # Detectar bordes usando el algoritmo Canny
    bordes = cv2.Canny(pista, 50, 150)
    return bordes

def detect_traffic_light(image):
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

    return image,color

while True:
    car2.set_velocity_and_request_state(
        forward=forward,
        turn=turn,
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=True,
        brakeSignal=False,
        reverseSignal=False
    )

    # Manejar el input del teclado para controlar la velocidad y el giro
    if keyboard.is_pressed('w'):
        forward = 0.3
    elif keyboard.is_pressed('s'):
        forward = -0.3
    else:
        forward = 0

    if keyboard.is_pressed('a'):
        if turn > -0.3:
            turn -= 0.15
    elif keyboard.is_pressed('d'):
        if turn < 0.3:
            turn += 0.15
    else:
        turn = 0
    
    if keyboard.is_pressed('g'):
        image_capture=1-image_capture

    if (image_capture):
        # Getting images from the different cameras 
        x, camera_image1 = car2.get_image(camera=car2.CAMERA_RGB)
        imagen = cv2.resize(camera_image1, (320, 240))
        imagen=imagen[120:240,0:320]
        #camera_image1,color=detect_traffic_light(camera_image1)
        #print("Angulo de giro:",turn)
        #cv2.imshow('Result', camera_image1)
        #cv2.waitKey(1)
        pista_segmentada = segmentar_pista(imagen)
        cv2.imwrite('ImagenSegmentada.jpg',pista_segmentada)
        cv2.imwrite('Camaras/Front/CamaraFrontX{:0.1f}_{}.jpg'.format(turn,i),pista_segmentada)
    i+=1   


