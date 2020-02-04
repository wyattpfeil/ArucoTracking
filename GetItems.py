import numpy as np
import time
import cv2
import cv2.aruco as aruco
import pickle
import asyncio
import websockets
import math

foundItemIds = []

itemCorrespondance = ["None", "None", "Bubble Gum","Rubik's Cube", "Tissues"]

xPos = None

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

message = "Checking out"
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
            # Checks if a matrix is a valid rotation matrix.
        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '
            currentId = ids[i][0]
            if currentId not in foundItemIds:
                foundItemIds.append(currentId)
        cv2.putText(frame, message, (0, 64), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, message, (0, 64), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        foundItemIds.sort()
        if len(foundItemIds) is not 0:
            message = "You've purchased " + str(len(foundItemIds)) + " item(s). "
            for i in foundItemIds:
                print(itemCorrespondance[i])
                message = message + itemCorrespondance[i] + ", "
        else:
            message = "You haven't purchased anything!"
        foundItemIds = []

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
