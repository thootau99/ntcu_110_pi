    # https://towardsdatascience.com/finding-driving-lane-line-live-with-opencv-f17c266f15db
# Testing edge detection for maze
import cv2
import numpy as np
import math
def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)], fit_fn

def clean_lines(lines, threshold):
    slope=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            k=(y2-y1)/(x2-x1)
            slope.append(k)
    #slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
          slope.pop(idx)
          lines.pop(idx)
        else:
          break

def imageDegreeCheck(image, it):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 7
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    low_threshold = 20
    high_threshold = 50

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # create a mask of the edges image using cv2.filpoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 1

    # define the Region of Interest (ROI) - source code sets as a trapezoid for roads
    imshape = image.shape

    vertices = np.array([[(0,imshape[0]),(100, 420), (1590, 420),(imshape[1],imshape[0])]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # mybasic ROI bounded by a blue rectangle

    #ROI = cv2.rectangle(image,(0,420),(1689,839),(0,255,0),3)

    # define the Hough Transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30 #minimum number of pixels making up a line
    max_line_gap = 200    # maximum gap in pixels between connectable line segments

    # make a blank the same size as the original image to draw on
    line_image = np.copy(image)*0

    # run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    left_lines = []
    right_lines = []
    try:
        if lines == None:
            return image,0
    except:
        pass
    for line in lines:
            for x1,y1,x2,y2 in line:

                angle = math.atan2(x2-x1, y2-y1)
                angle = angle * 180 / 3.14
                intangle = int(angle)
                if (intangle < 170 and intangle > 135) or (intangle < 50 and intangle > 23):
                    if intangle == 180:
                        continue
                    if intangle < 50 and intangle > 23:
                        right_lines.append(line)
                    else:
                        left_lines.append(line)
                        # intangle = abs(intangle - 180)
                    # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

                # if (angle < 145 and angle > 100) or (intangle < 50 and intangle > 28):
                #     # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                #     left_angle_av = left_angle_av + angle
                #     left_count = left_count + 1
                
                # if (angle < 160 and angle > 145) or (intangle < 50 and intangle > 28):
                #     # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                #     right_angle_av = right_angle_av + angle
                #     right_count = right_count + 1
                #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                #     cv2.putText(line_image, str(c), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA, True)

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
    
    try:
        left_vtx, left_fun = calc_lane_vertices(left_points, 100, image.shape[0])
    except :
        print("can't find line left")
        return image, "rc 10 0 0 0"
    try:
        right_vtx, right_fun = calc_lane_vertices(right_points, 100, image.shape[0])
    except :
        print("can't find line right")
        return image, "rc -10 0 0 0"
    

    leftLineCenter, rightLineCenter = int(left_fun(image.shape[0]/2)),int(right_fun(image.shape[0]/2))
    center = image.shape[1] / 2
    print(leftLineCenter, rightLineCenter,center)
    lineav = (leftLineCenter + rightLineCenter) / 2
    leftDegree = math.atan2(left_vtx[1][0]-left_vtx[0][0], left_vtx[1][1] - left_vtx[0][1]) * 180 / 3.14
    rightDegree = math.atan2(right_vtx[1][0]-right_vtx[0][0], right_vtx[1][1] - right_vtx[0][1]) * 180 / 3.14
    print(leftDegree, rightDegree)
    # print(int(abs(abs(leftDegree) - rightDegree)))
    # print( abs(int(abs(leftDegree)) - int(abs(rightDegree))), abs(int(abs(rightDegree)) - int(abs(leftDegree))) )
    degree = abs(abs(leftDegree) - abs(rightDegree))
    result = ""
    leftcentered = False
    rightcentered = False

    if abs(int(abs(leftDegree)) - int(abs(rightDegree))) >= 4:
        if abs(abs(int(leftDegree)) - 40) > 5:
            print("qua right turn left")
            result = "rc 10 0 0 0"
            return image, result

        if abs(abs(int(rightDegree)) - 45) > 3:
            print("qua left turn right")
            result = "rc -10 0 0 0"
            return image, result

    elif abs(int(abs(leftDegree)) - int(abs(rightDegree))) < 3:
        # result = "cw 0"
        if leftLineCenter > (center-50):
            result = "cw 3"
        else:
            leftcentered = True

        if rightLineCenter < (center+50):
            result = "cw -3"
        else:
            rightcentered = True
        # if abs(abs(int(leftDegree)) - 40) > 5:
        #     print("qua right turn left")
        #     result = "cw 3"
        #     return image, result
        # else:
        #     result = "rc 0 0 0 0"
        #     leftcentered = True

        # if abs(int(rightDegree)) - 45 > 5:
        #     print("qua left turn right")
        #     result = "rc -10 0 0 0"
        #     return image, result
        # else:
        #     result = "rc 0 0 0 0"
        #     return image, result
            # result = "cw "+ str(3)
        # result = "cw "+ str(int(abs(degree - 10)))
    # if abs(int(abs(rightDegree)) - int(abs(leftDegree))) > 5:
    #     print("qua right turn left")
    #     result = "rc- 10 0 0 0"
    #     return image, result
    #     # result = "cw "+ str(-3)
    # else:
    #     rightcentered = True

    if leftcentered and rightcentered:
        result = "cw 0"
        return image, result

        # result = "cw "+ str(-int(abs(degree - 10)))
    # cv2.line(image, left_vtx[0], left_vtx[1], (255,0,0), 10)
    # cv2.line(image, right_vtx[0], right_vtx[1], (255,0,0), 10)
    # if left_count == 0:
    #     angle_av = right_angle_av / right_count
    # else:
    #     angle_av = left_angle_av / left_count
    #     left = True
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    # cv2.startWindowThread()
    # cv2.namedWindow("ppp")
    # cv2.imshow("ppp", image)
    # cv2.waitKey(1)
    # if not left:
    #     print(angle_av-(angle_av + flipav)/2)
    # else:
    #     print(abs(angle_av-(angle_av + flipav)/2))
    # draw the line on the original image
    return image, result
