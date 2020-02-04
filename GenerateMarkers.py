import numpy as np
import cv2
import cv2.aruco as aruco

# Select type of aruco marker (size)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
i = 20
img = aruco.drawMarker(aruco_dict, i, 500)
cv2.imwrite(str(i) + "_marker.png", img)