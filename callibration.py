import numpy as np
import time
import cv2
import cv2.aruco as aruco
import pickle

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=0.04,
    markerSeparation=0.01,
    dictionary=aruco_dict)

mtx = None
dist = None

cam = cv2.VideoCapture(4)

time.sleep(5)

# Read an image or a video to calibrate your camera
# I'm using a video and waiting until my entire gridboard is seen before calibrating
# The following code assumes you have a 5X7 Aruco gridboard to calibrate with
while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
        # Make sure markers were detected before continuing
        if ids is not None and corners is not None and len(ids) > 0 and len(corners) > 0 and len(corners) == len(ids):
            # The next if makes sure we see all matrixes in our gridboard
            if len(ids) == len(board.ids):
                # Calibrate the camera now using cv2 method
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objectPoints=board.objPoints,
                        imagePoints=corners,
                        imageSize=gray.shape, #[::-1], # may instead want to use gray.size
                        cameraMatrix=None,
                        distCoeffs=None)


                 # Print matrix and distortion coefficient to the console
                break
print("Callibrated!")
print(dist)
print("")
print(mtx)
callibrationData = {"dist": dist, "mtx": mtx}
with open('callibrationData.pickle', 'wb') as handle:
    pickle.dump(callibrationData, handle, protocol=pickle.HIGHEST_PROTOCOL)