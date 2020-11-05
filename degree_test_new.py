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

def select_white_yellow(image):
    # yellow color mask
    lower = np.uint8([ 26,  43, 46])
    upper = np.uint8([ 34, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    # mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = yellow_mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def imageDegreeCheck(image, it):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    low_yellow = np.array([21, 39 ,64]) # 15, 90 ,140
    high_yellow = np.array([40,255,255]) # 50, 160 ,255
    mask_hsv = select_white_yellow(hsv)
    kernel_size = 7
    gray = convert_gray_scale(mask_hsv)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    low_threshold = 100
    high_threshold = 300

    edges = cv2.Canny(blur_gray, 50, 150)
    masked_edges = select_region(edges)



    cv2.startWindowThread()
    cv2.namedWindow("123")
    cv2.imshow("123", masked_edges)
    cv2.waitKey(1)

    # make a blank the same size as the original image to draw on
    line_image = np.copy(image)*0

    # run Hough on edge detected image
    lines = hough_lines(masked_edges)
    left_lines = []
    right_lines = []
    try:
        if lines == None:
            return image,0
    except:
        pass
    for line in lines:
            for x1,y1,x2,y2 in line:
                # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                if x1 == x2:
                    continue
                angle = math.atan2(x2-x1, y2-y1)
                angle = angle * 180 / 3.14
                intangle = int(angle)
                if (intangle < 178 and intangle > 135) or (intangle < 80 and intangle > 5):
                    # print(intangle)
                    if intangle == 180:
                        continue
                    if intangle < 50 and intangle > 5:
                        right_lines.append(line)
                    else:
                        left_lines.append(line)
                        # intangle = abs(intangle - 180)

                # if (angle < 145 and angle > 100) or (intangle < 50 and intangle > 28):
                #     # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                #     left_angle_av = left_angle_av + angle
                #     left_count = left_count + 1
                # if (angle < 160 and angle > 145) or (intangle < 50 and intangle > 28):
                #     # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                #     right_angle_av = right_angle_av + angle
                #     right_count = right_count + 1
                #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                    cv2.putText(line_image, str(intangle), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA, True)

    cv2.startWindowThread()
    cv2.namedWindow("ppaaaap")
    cv2.imshow("ppaaaap", line_image)
    cv2.waitKey(1)

    clean_lines(left_lines, 0.4)
    clean_lines(right_lines, 0.4)
    left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    l = lines_edges
    cv2.startWindowThread()
    cv2.namedWindow("ppp_hsv")
    cv2.imshow("ppp_hsv", l)
    cv2.waitKey(1)
    try:
        left_vtx, left_fun = calc_lane_vertices(left_points, 200, image.shape[0])
    except :
        print("can't find line left")
        return image, "rc 10 0 0 0"
        
    try:
        right_vtx, right_fun = calc_lane_vertices(right_points, 200, image.shape[0])
    except :
        print("can't find line right")
        l = edges
        cv2.startWindowThread()
        cv2.namedWindow("ppp")
        cv2.imshow("ppp", l)
        cv2.waitKey(1)
        return image, "rc -10 0 0 0"
    

    cv2.line(image, left_vtx[0], left_vtx[1], (255,0,0), 10)
    cv2.line(image, right_vtx[0], right_vtx[1], (255,0,0), 10)
    
    l = image
    cv2.startWindowThread()
    cv2.namedWindow("ppp123")
    cv2.imshow("ppp123", l)
    cv2.waitKey(1)

    leftLineCenter, rightLineCenter = int(left_fun(image.shape[0]/2)),int(right_fun(image.shape[0]/2))
    center = image.shape[1] / 2
    print(leftLineCenter, rightLineCenter,center)
    lineav = (leftLineCenter + rightLineCenter) / 2
    leftDegree = math.atan2(left_vtx[1][0]-left_vtx[0][0], left_vtx[1][1] - left_vtx[0][1]) * 180 / 3.14
    rightDegree = math.atan2(right_vtx[1][0]-right_vtx[0][0], right_vtx[1][1] - right_vtx[0][1]) * 180 / 3.14
    # print(leftDegree, rightDegree)
    # print(int(abs(abs(leftDegree) - rightDegree)))
    # print( abs(int(abs(leftDegree)) - int(abs(rightDegree))), abs(int(abs(rightDegree)) - int(abs(leftDegree))) )
    degree = abs(abs(leftDegree) - abs(rightDegree))
    result = ""
    leftcentered = False
    rightcentered = False
    print(leftDegree, rightDegree)
    print(abs(int(abs(leftDegree)) - int(abs(rightDegree))))

    if abs(int(abs(leftDegree)) - int(abs(rightDegree))) >= 4:
        if leftDegree > 3:

            print("can't get actually left line so turn left")
            result = "rc -10 0 0 0"
            return image, result
        if rightDegree < -3:
            print("can't get actually right line so turn right")
            result = "rc 10 0 0 0"
            return image, result
        print(abs(int(leftDegree) - (-11)))
        if abs(int(leftDegree) - (-11)) > 5:
            print("qua leftt turn right")
            result = "rc 10 0 0 0"
            return image, result

        if abs(abs(int(rightDegree)) - 18) > 10:
            print("qua right turn left")
            result = "rc -10 0 0 0"
            return image, result

    elif abs(int(abs(leftDegree)) - int(abs(rightDegree))) < 3:
        
        # result = "cw 0"
        if leftLineCenter > (center-5):
            result = "cw 3"
        else:
            leftcentered = True

        if rightLineCenter < (center+5):
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
        print(result)

        return image, result
    print(result)
        # result = "cw "+ str(-int(abs(degree - 10)))
    # if left_count == 0:
    #     angle_av = right_angle_av / right_count
    # else:
    #     angle_av = left_angle_av / left_count
    #     left = True
    # if not left:
    #     print(angle_av-(angle_av + flipav)/2)
    # else:
    #     print(abs(angle_av-(angle_av + flipav)/2))
    # draw the line on the original image
    return image, result


# im = cv2.imread("nnss.png")

# im, result = imageDegreeCheck(im, True)


cap = cv2.VideoCapture('output.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    im, result = imageDegreeCheck(frame, True)
    cv2.imshow('frame',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()