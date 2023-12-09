import cv2 as cv
from cv2 import aruco
import numpy as np

calibrated_data_path = "calibrated_data/MultiMatrix.npz"  # Path to the calibrated data

calibrated_data = np.load(calibrated_data_path)
print(calibrated_data.files)

cam_mat = calibrated_data["camMatrix"]
dist_coef = calibrated_data["distCoef"]
r_vectors = calibrated_data["rVector"]
t_vectors = calibrated_data["tVector"]

MARKER_SIZE = 4  # size in cm # table markers size
MARKER_SIZE2 = 5  # size in cm # gamma camera marker size

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

    # For Table Markers
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
                f"X:{round(tVec[i][0][0],1)} Y:{round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            )

    # Gamma Marker
    if marker_corners2:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners2, MARKER_SIZE2, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs2.size)
        for ids, corners, i in zip(marker_IDs2, marker_corners2, total_markers):
            cv.polylines(               
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)

            # Extracting the top-right corner coordinates
            top_right = corners[0].ravel()

            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            # Draw the pose of the marker with a 90-degree rotation around the Y-axis
            rotated_rVec = cv.Rodrigues(np.array([0, np.radians(90), 0]))[0]
            rotated_point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rotated_rVec, tVec[i], 4, 4)
#################
            # Define the corners of the artificial plane with the specified dimensions
            plane_size = MARKER_SIZE  # Dimensions of the plane in mm, using the marker size
            plane_corners = np.array([
                [-plane_size/2, 0, -plane_size/2],  # Top-left corner
                [plane_size/2, 0, -plane_size/2],   # Top-right corner
                [plane_size/2, 0, plane_size/2],    # Bottom-right corner
                [-plane_size/2, 0, plane_size/2],   # Bottom-left corner
            ], dtype=np.float32)

            # Project the corners onto the image plane
            image_points, _ = cv.projectPoints(plane_corners, rotated_rVec, tVec[i], cam_mat, dist_coef)

            # Draw the artificial plane (rectangle) on the frame
            image_points = image_points.astype(int)
            #cv.drawContours(frame, [image_points], -1, (0, 0, 0), -1)  # Use -1 for thickness to fill the rectangle with black
            cv.drawContours(frame, [image_points], -1, (0, 165, 255), -1)  # (0, 165, 255) corresponds to light orange in BGR
#############
            # Extracting the rotated X, Y coordinates
            rotated_X = rotated_point[0, 0]
            rotated_Y = rotated_point[0, 1]

            # Printing the ID, distance, and rotated coordinates
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
                f"Rotated X:{round(rotated_X[0], 1)} Rotated Y:{round(rotated_Y[0], 1)} ",
                #f"Rotated X:{round(rotated_X, 1)} Rotated Y:{round(rotated_Y, 1)} ",
                (top_right[0], top_right[1] + 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv.LINE_AA,
            )

    # Display the frame
    cv.imshow("frame", frame)
    
    # Check for the 'q' key to quit the loop
    key = cv.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture object and close the windows
cap.release()
cv.destroyAllWindows()