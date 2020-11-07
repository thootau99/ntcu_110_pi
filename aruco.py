import numpy as np
import cv2
import sys
import math
from scipy.spatial.transform import Rotation
aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = aruco.DetectorParameters_create()

parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

cc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', cc, 20.0, (480, 360))

def drawCenter(frame, corners, w, h):
    if self.TargetID not in seen_id_list:
        pass
    else:
        cx = int((corners[ind][0][0][0]+corners[ind][0][1][0]+corners[ind][0][2][0]+corners[ind][0][3][0])/4)
        cy = int((corners[ind][0][0][1]+corners[ind][0][1][1]+corners[ind][0][2][1]+corners[ind][0][3][1])/4)
        cv2.line(frame, (int(w/2), int(h/2)), (cx, cy), (0,255,255), 3)

    return frame

def rotationVectorToEulerAngles(rvec):
    r = Rotation.from_rotvec(rvec)
    return r.as_euler('xyz', degrees=True)

def aru(frame):
    h, w = frame.shape[:2]
    cnt=0
    marker_length = 0.1
    camera_matrix = np.array([[974.9035727,    0,         173.4483866, ],
 [  0,         906.94858818, 241.45362918,],
 [  0,           0,           1        ]])

    distortion_coeff = np.array( [[-5.60512132e-01,  5.57573002e+00, -8.93361527e-02, -1.13456082e-03,
 -4.28452394e+01]] )
    ids=[]
    try:
        x = []
        y = []
        z = []
        d = []
        yaw = []
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
                eular_angle = rotationVectorToEulerAngles(rvec)
                print(eular_angle)
                # euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
                # print("x : " + str(tvec[0]))
                # print("y : " + str(tvec[1]))
                # print("z : " + str(tvec[2]))
                # print("rollcm : " + str(euler_angle[0]))
                # print("pitch: " + str(euler_angle[1]))
                x.append(tvec[0])
                y.append(tvec[1])
                z.append(tvec[2])
                d.append(tvec[2] * 24)
                yaw.append(euler_angle[0])
                # print("%.1f cm -- %.0f deg" % ((tvec[2] * 2.5), (rvec[2] / math.pi * 180)))

                draw_pole_length = marker_length/2 # 現実での長さ[m]
                aruco.drawAxis(frame, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)
                

        out.write(frame)
        cnt+=1

        key = cv2.waitKey(50)
        if key == 27: # ESC
            return
    finally:
        try:
            if ids == None:
                ids = []
                x = []
                y = []
                z = []
                d = []
        except:
            pass
        try:
            return x, y, z, d, yaw,ids,frame
        except:
            return [], [],[], [], [], [], frame
 
    if __name__ == "__main__":
        main()
