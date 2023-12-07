import cv2 as cv
from cv2 import aruco
import numpy as np

calibrated_data_path = "calibrated_data/MultiMatrix.npz" #Path to the calibrated data

calibrated_data = np.load(calibrated_data_path)
print(calibrated_data.files)

cam_mat = calibrated_data["camMatrix"]
dist_coef = calibrated_data["distCoef"]
r_vectors = calibrated_data["rVector"]
t_vectors = calibrated_data["tVector"]

MARKER_SIZE = 4  # size in cm #table markers size
MARKER_SIZE2 = 5  # size in cm #gamma camera marker size

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
marker_dict2 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    marker_corners2, marker_IDs2, reject = aruco.detectMarkers(
        gray_frame, marker_dict2, parameters=param_markers
    )

#For Table Markers
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"ID: {ids[0]} D: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"X:{round(tVec[i][0][0],1)} Y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            )
#Gamma Marker
    if marker_corners2:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners2, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs2.size)
        for ids, corners, i in zip(marker_IDs2, marker_corners2, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"ID: {ids[0]} D: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"X:{round(tVec[i][0][0],1)} Y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            ) 
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()