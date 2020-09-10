import numpy as np
import cv2 
import sys
import math
aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = aruco.DetectorParameters_create()

parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

def aru(frame):
    cnt=0
    marker_length = 0.1
    camera_matrix = np.array([[974.9035727,    0,         173.4483866, ],
 [  0,         906.94858818, 241.45362918,],
 [  0,           0,           1        ]])

    distortion_coeff = np.array( [[-5.60512132e-01,  5.57573002e+00, -8.93361527e-02, -1.13456082e-03,
 -4.28452394e+01]] )

    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)

        aruco.drawDetectedMarkers(frame, corners, ids, (0,255,255))
        x = []
        y = []
        z = []
        d = []
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
                # print("x : " + str(tvec[0]))
                # print("y : " + str(tvec[1]))
                # print("z : " + str(tvec[2]))
                # print("rollcm : " + str(euler_angle[0]))
                # print("pitch: " + str(euler_angle[1]))
                x.append(tvec[0])
                y.append(tvec[1])
                z.append(tvec[2])
                d.append(tvec[2] * 2.4)
                # print("%.1f cm -- %.0f deg" % ((tvec[2] * 2.5), (rvec[2] / math.pi * 180)))
                
                draw_pole_length = marker_length/2 # 現実での長さ[m]
                aruco.drawAxis(frame, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

        cv2.imshow('fa', frame)
        cnt+=1

        key = cv2.waitKey(50)
        if key == 27: # ESC
            return
    finally:
        try:
            if ids == None:
                ids = []
        except:
            pass
        return x, y, z, d, ids

    if __name__ == "__main__":
        main()