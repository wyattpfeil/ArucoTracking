import numpy as np
import time
import cv2
import cv2.aruco as aruco
import pickle
import asyncio
import websockets
import math

xPos = None

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
     markersX=5,
     markersY=7,
     markerLength=0.04,
     markerSeparation=0.01,
     dictionary=aruco_dict)

mtx = None
dist = None

with open('callibrationData.pickle', 'rb') as handle:
    callibrationData = pickle.load(handle)
    dist = callibrationData["dist"]
    mtx = callibrationData["mtx"]


cap = cv2.VideoCapture(4)

time.sleep(2)
while (True):
    ret, frame = cap.read()
    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.05, mtx, dist)
        # (rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)
            xPos = tvec[0][0][0]
            yPos = tvec[0][0][1]
            zPos = tvec[0][0][2]

            rmat = cv2.Rodrigues(rvec[i])[0]
            yawpitchroll_radian = -rotationMatrixToEulerAngles(rmat)
            yawpitchroll_radian[0] = yawpitchroll_radian[0]-math.pi
            yawpitchroll_angles = 180*yawpitchroll_radian/math.pi
            xRotation = yawpitchroll_angles[2]
            yRotation = yawpitchroll_angles[1]
            zRotation = yawpitchroll_angles[0]
            
            if(xRotation < 0):
                xRotation = 360 - abs(xRotation)
            if(yRotation < 0):
                yRotation = 360 - abs(yRotation)
            if(zRotation < 0):
                zRotation = 360 - abs(zRotation)
            #print("x: " + str(xRotation))
            print(zRotation)
            # Checks if a matrix is a valid rotation matrix.
        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '
            currentId = ids[i][0]

       # cv2.putText(frame, message, (0, 64), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found
        #cv2.putText(frame, message, (0, 64), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()