# region: package imports
import os
import cv2
import time
import keyboard
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
def sobel(image,thresh_min = 30,thresh_max=100):
    #image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)

     # 3) Take the absolute value of the derivative or gradient
    absobelx = np.absolute(sobelx)

    #print(type(absobelx))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_x = np.uint8(255*absobelx/np.max(absobelx))

     # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    sx_binary = np.zeros_like(scaled_x)

    sx_binary[(scaled_x>=thresh_min)&(scaled_x<=thresh_max)]=255
    #cv2.imshow("sobelx",sx_binary)
    #print(sx_binary)
    return sx_binary



def perspective(image):
    h,w = image.shape[:2]
    #cv2.circle(image,(300,385),5,(0,0,255),-1)
    #cv2.circle(image,(500,341),5,(0,0,255),-1)
    #cv2.circle(image,(40,580),5,(0,0,255),-1)
    #cv2.circle(image,(800,450),5,(0,0,255),-1)
    pts1 = np.float32([[310,385],[500,385],[0,580],[650,580]])
    pts2 = np.float32([[0,0],[600,0],[0,600],[600,600]])
    #cv2.imshow('image',image)

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    minv = cv2.getPerspectiveTransform(pts2,pts1)

    result = cv2.warpPerspective(image,matrix,(650,650),flags = cv2.INTER_LINEAR)
    #cv2.imshow('image',result)
    return result ,minv

def hls_select(image):
    hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]

    thresh_s = (110, 255)
    Sbinary = np.zeros_like(l)
    Sbinary[(s > thresh_s[0]) & (s <= thresh_s[1])] = 255
    #cv2.imshow("S_binary",Sbinary)
    return h,l,s,Sbinary

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram

def find_lane(per):
    out_image = np.dstack((per,per,per))*255
    #cv2.imshow("outimage",out_img)

    #for final representaion stack the perseptive image
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int_(histogram.shape[0]//2)
    #print(midpoint)
    #np.argmax give indices of maximum value
    leftx_base = np.argmax(histogram[:midpoint])
    #print(leftx_base)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print(rightx_base)
    '''Set up windows and window hyperparameters
    Our next step is to set a few hyperparameters related to our sliding windows,
    and set them up to iterate across the binary activations
    in the image. We have some base hyperparameters below, but don't forget to
    try out different values in your own implementation to see what works best!'''
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape means rake perspectivw image and then devide by no. of windows
    window_height = np.int_(per.shape[0]//nwindows)
    #print(window_height)
    #we get 66 pixel

    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    #non zeros means pixel which have some amount of intensity
    nonzero = per.nonzero()
    #print(nonzeros)
    nonzeroy = np.array(nonzero[0])
    #print(nonzeroy)
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    #previsouly we find out left and right bse coresponsding to x axis now we have to update or use forfuthrt update there fore we are assinging to
    # another variable so that oroginal base doesn't change
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    #the location of an item in an array.(or index)
    #we used left_lane_inds and right_lane_indsto hold the pixel values contained within the boundaries of a given sliding window
    left_lane_inds = []
    right_lane_inds = []

     # Step through the windows one by one

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        #lower box at y axis
        win_y_low = per.shape[0] - (window+1)*window_height
        win_y_high = per.shape[0] - window*window_height
        #print("win_y_high",win_y_high)
        #position of window at x axis at left side
        win_xleft_low = leftx_current - margin
        #print(win_xleft_low)
        win_xleft_high = leftx_current + margin
        win_xright_low =  rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_image,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_image,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        #cv2.imshow("outout",out_image)
         # Identify the nonzero pixels in x and y axis within the window
         # previous we found all non zeros number now we have to find non zeros in box which we made
         #and that can be done by and operation and save into list at 0 indices
        good_left_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        #print(good_left_inds)

    ########

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        #base will update in when ther e change in center or say pixels which no include in min pix region
        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        #print(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Extract left and right line pixel positions

    leftx = nonzerox[left_lane_inds]

    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty,leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
    except :
        pass
# Generate x and y values for plotting
    '''numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].

    The endpoint of the interval can optionally be excluded.'''
    ploty = np.linspace(0, per.shape[0]-1, per.shape[0] )
    #print(ploty.dtype)
    # Assuming we have `left_fit` and `right_fit` from `np.polyfit` before
    # Generate x and y values for plotting
    try :
        left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:

            # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    ## Visualization ##
    # Colors in the left and right lane regions
    out_image[lefty, leftx] = [255, 0, 0]
    out_image[righty, rightx] = [0, 0, 255]
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='blue')

    # Plots the left and right polynomials on the lane lines
##    plt.plot(left_fitx, ploty, color='yellow')
    #plt.show()
    cv2.imshow("out",out_image)
##    plt.plot(right_fitx, ploty, color='yellow')
##    plt.imshow(out_image)
##    plt.show()



    #plt.imshow(out_image)
    #plt.show()


    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds,out_image
def draw_lines(img, left_fit, right_fit,out_image):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)
    lane_img = np.copy(img)
    #print(color_warp.shape)
    #print(out_image.shape)

    # Find left and right points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp,np.int32([pts_left]),False,(255,0,0),15)
    cv2.polylines(color_warp,np.int32([pts_right]),False,(0,0,255),15)

    #cv2.addWeighted(out_image, 1, color_warp, 0.3, 0)
    #cv2.imshow("asd",color_warp)
    #print(color_warp.shape)

    # Warp the blank back to original image space using inverse perspective matrix
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    #print(unwarp_img.shape)
    cv2.imshow("unwrap image",unwarp_img)
    lane_image = cv2.warpPerspective(out_image, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

    result = cv2.addWeighted(lane_img, 0.5, lane_image, 200, 0)
    #cv2.imshow("drawline",result)
    result = cv2.addWeighted(img,0.9, unwarp_img, 0.5, 0)
    cv2.imshow("drawline",result)
    return result
def measure_curvature(image, left_fit, right_fit,xmtr_per_pixel, ymtr_per_pixel):
    # Define conversions in x and y from pixels space to meters

    ploty = np.linspace(0,image.shape[0]-1,image.shape[0])
    # Find left and right points.
    # Assuming we have `left_fit` and `right_fit` from `np.polyfit` before
    # Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
       # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    '''# Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space'''
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)

    '''
    NOW
    Calculates the curvature of polynomial functions in meters.
    Calculation of R_curve (radius of curvature)
    Rcurve = ((1+(2*A*y+B)**2)**3/2)/ (|2*A|))
    Y = y_value*ymtr_per_pixel for left
    '''

    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_rad , right_rad
def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ## Image mid horizontal position
    #xmax = img.shape[1]*xmtr_per_pixel
    ymax = img.shape[0]*ymtr_per_pixel

    center = img.shape[1] / 2
    '''
        off-center distance calculation, based on 2 assumptions:
            1) camera mounted in the center of vehicle
            2) road lane is 3.7 meters wide
    '''

    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]

    mid = lineLeft + (lineRight - lineLeft)/2
    #print(mid)
    dist = (mid - center) * xmtr_per_pixel


    if dist > 0.05 :
        message = 'Vehicle location: left'.format(dist)
        Turn = 'Turn RIGHT'
    elif (dist < -0.15):

        message = 'Vehicle location:  right '.format(abs(dist))
        Turn = ''

    else:
        #message = 'Vehicle location: {:.2f} m left '.format(abs(dist))
        Turn = ''
        message = 'AT center'
    return message ,Turn ,mid
def show_curvatures(img, left_fit, right_fit,xmtr_per_pixel,ymtr_per_pixel):
    (left_curvature, right_curvature) = measure_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt , Turn,mid = dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    #out_img = cv2.line(out_img, (int(mid+50),int(mid+300)), (int(mid+50),int(mid+100)), (0,255,0), 5)
    cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
##    cv2.putText(out_img, 'left lane curvature: {:.2f} m'.format(left_curvature),
##                (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out_img, Turn , (300, 500), cv2.FONT_HERSHEY_DUPLEX , 1, (255,0,0), 2)

    return out_img



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

# Start spawn model
QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

forward = 0
turn = 0
image_capture=0
frame_counter = 0

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
        forward = 0.5
    elif keyboard.is_pressed('s'):
        forward = -0.5
    else:
        forward = 0
    if keyboard.is_pressed('a'):
        turn = turn-0.3
    elif keyboard.is_pressed('d'):
        turn = turn+0.3
    else:
        turn = 0
    # # Getting images from the different cameras 
    x, image = car2.get_image(camera=car2.CAMERA_RGB)
    tic = time.time()
    frame_counter += 1
    #If the last frame is reached, reset the capture and the frame_counter
    # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     frame_counter = 0 #Or whatever as long as it is the same as next line
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    resize = cv2.resize(image,(800,600))

    blur = cv2.GaussianBlur(resize,(5,5),0)
    per,minv = perspective(blur)
    #blur =  cv2.resize(blur,(480,360))
    h,l,s,lbinary= hls_select(per)
    combine_binary = np.zeros_like(lbinary)
    sobelx = sobel(per)

    combine_binary[(sobelx >=1) | (lbinary >=1)] = 255
    # cv2.imshow("Combine_binary",combine_binary)

    color_binary = np.dstack(( np.zeros_like(sobelx), sobelx, lbinary))
    #per,minv = perspective(combine_binary)
    histogram = hist(combine_binary)
    # plt.plot(histogram)
    # plt.show()
    imshow = cv2.imshow("color_binary",image)
    cv2.waitKey(1)
    
    left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds,out_image = find_lane(combine_binary)
    #else:

    image_inv = draw_lines(blur,left_fit,right_fit,out_image)
    xmtr_per_pixel = 3.7/800
    ymtr_per_pixel = 30/600
    #print(image_inv.shape)
    #result = measure_curvature(image_inv,left_fit,right_fit)
    result = show_curvatures(image_inv,left_fit,right_fit,xmtr_per_pixel,ymtr_per_pixel)

    #plt.imshow(image_inv)
    #plt.show()

    #cv2.imshow("sda",out_image)
    #plt.show()
    #print(image_inv.shape)
    toc = time.time()
    fps = 1/(toc-tic)
    print("Estimated frames per second : {0}".format(int(fps)))

    cv2.imshow("final",result)
    #cv2.imshow("image_inv",image_inv)


    if cv2.waitKey(30) &0xFF== ord('q'):

        cv2.destroyAllWindows()
        break

 


