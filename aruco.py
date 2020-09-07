import numpy as np
import cv2 
import sys
import math
aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = aruco.DetectorParameters_create()

parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

def main():
    cap = cv2.VideoCapture(0)
    cnt=0
    marker_length = 0.1
    camera_matrix = np.array([[9.31357583e+03, 0.00000000e+00, 1.61931898e+03],
[0.00000000e+00, 9.64867367e+03, 1.92100899e+03],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    distortion_coeff = np.array( [[ 0.22229833, -6.34741982,  0.01145082,  0.01934784, -8.43093571]] )

    try:
        while True:
            ret, frame = cap.read()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)

            aruco.drawDetectedMarkers(frame, corners, ids, (0,255,255))

            if len(corners) > 0:
                for i, corner in enumerate(corners):
                    # rvec -> rotation vector, tvec -> translation vector
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)

                    tvec = np.squeeze(tvec)
                    rvec = np.squeeze(rvec)
                    rvec_matrix = cv2.Rodrigues(rvec)
                    rvec_matrix = rvec_matrix[0]
                    transpose_tvec = tvec[np.newaxis, :].T
                    proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                    euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
                    print("x : " + str(tvec[0]))
                    print("y : " + str(tvec[1]))
                    print("z : " + str(tvec[2]))
                    print("roll : " + str(euler_angle[0]))
                    print("pitch: " + str(euler_angle[1]))

                    print("%.1f cm -- %.0f deg" % ((tvec[2] * 4.2), (rvec[2] / math.pi * 180)))
                    draw_pole_length = marker_length/2 # 現実での長さ[m]
                    aruco.drawAxis(frame, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

            cv2.imshow('fa', frame)
            cnt+=1

            key = cv2.waitKey(50)
            if key == 27: # ESC
                break
    finally:
        cap.release()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass