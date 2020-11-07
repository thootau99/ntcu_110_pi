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
    lower = np.uint8([ 25,  37, 156])
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
    top_left     = [cols*0.2, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.8, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def imageDegreeCheck(image, mode):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask_hsv = select_white_yellow(hsv)
    kernel_size = 7
    gray = convert_gray_scale(mask_hsv)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    low_threshold = 100
    high_threshold = 300
    edges = cv2.Canny(blur_gray, 50, 150)
    masked_edges = select_region(edges)


    cv2.startWindowThread()
    cv2.namedWindow("456")
    cv2.imshow("456", mask_hsv)
    cv2.waitKey(1)

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
    left_lines_start = []
    right_lines_start = []
    left_lines_start_mean = 0
    right_lines_start_mean = 0
    try:
        if lines == None:
            return image,0, ""
    except:
        pass

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            # if x2==x1:
            #     continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            cv2.putText(line_image, str(slope), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA, False)

            if slope < 0: # y is reversed in image
                left_lines.append(line)
                # print(line)
                left_lines_start.append(x1)
            else:
                right_lines.append(line)
                right_lines_start.append(x1)
        left_lines_start_mean = np.mean(left_lines_start)
        right_lines_start_mean = np.mean(right_lines_start)
    count = 0
    for start in left_lines_start:
        s = right_lines_start_mean - start
        # print(s)
        if s < 10:
            # print("poping same left",s)
            index = left_lines_start.index(start)
            # print(index, left_lines, left_lines_start)
            try:
                right_lines_start.append(left_lines_start[index])
                right_lines.append(left_lines[index])
                left_lines_start.pop(index)
                left_lines.pop(index)
            except:
                pass
    count = 0
    for start in right_lines_start:
        s = right_lines_start_mean - start
        # print(s)
        if s > 10:
            # print("poping same left",s)
            index = right_lines_start.index(start)
            # print(index, left_lines, left_lines_start)
            try:
                left_lines_start.append(right_lines_start[index])
                left_lines.append(right_lines[index])
                right_lines_start.pop(index)
                right_lines.pop(index)
            except:
                pass
    if len(left_lines) == 0 and len(right_lines) != 0:
        if len(right_lines) == 1:
            pass
        else:
            count = 0
            for index, right_mean in enumerate(right_lines_start):
                if right_mean - right_lines_start_mean < -50:
                    # print("switch right to left", right_mean, right_lines[index])
                    # if index == len(right_lines):
                    #     break
                    try:
                        
                        left_lines.append(right_lines[index-count])
                        right_lines.pop(index-count)
                        count = count + 1
                    except:
                        pass
    if len(right_lines) == 0 and len(left_lines) != 0:
        if len(left_lines) == 1:
            pass
        else:
            count = 0
            for index, left_mean in enumerate(left_lines_start):
                if left_mean - left_lines_start_mean > 50:
                    # print("switch right to left", left_mean, left_lines[index])
                    if index == len(left_lines):
                        break
                    try:
                        right_lines.append(left_lines[index-count])
                        left_lines.pop(index-count)
                        count = count + 1
                    except:
                        pass
    # print(left_lines, right_lines)
            
            

    cv2.startWindowThread()
    cv2.namedWindow("ppaaaap")
    cv2.imshow("ppaaaap", line_image)
    cv2.waitKey(1)

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
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
    status = ""
    result  = ""
    try:
        left_vtx, left_fun = calc_lane_vertices(left_points, 100, image.shape[0])
        leftDegree = math.atan2(left_vtx[1][0]-left_vtx[0][0], left_vtx[1][1] - left_vtx[0][1]) * 180 / 3.14
        left_slope = (left_vtx[1][1]-left_vtx[0][1])/(left_vtx[1][0]-left_vtx[0][0])
        cv2.line(image, left_vtx[0], left_vtx[1], (255,0,0), 10)
        cv2.putText(image, str(left_slope), (left_vtx[0][0], left_vtx[0][1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA, False)
        lx = left_vtx[1][0]
    except :
        status = "left"
        lx = 0
    try:
        right_vtx, right_fun = calc_lane_vertices(right_points, 100, image.shape[0])
        rightDegree = math.atan2(right_vtx[1][0]-right_vtx[0][0], right_vtx[1][1] - right_vtx[0][1]) * 180 / 3.14
        right_slope = (right_vtx[1][1]-right_vtx[0][1])/(right_vtx[1][0]-right_vtx[0][0])
        cv2.line(image, right_vtx[0], right_vtx[1], (255,0,0), 10)
        cv2.putText(image, str(right_slope), (right_vtx[0][0], right_vtx[0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA, False)

        rx = right_vtx[1][0]
    except :
        status = status + "right"
        rx = 0
        # return image, "rc -10 0 0 0", status
    if status == "left":
        if rightDegree < 33 and rightDegree > -4:
            status = "notrealleft"
            # print(rightDegree, status)

            result = "rc 10 0 0 0"
        else:
            status = "left"
            result = "rc -10 0 0 0"
        return image, result, status
    elif status == "right":
        if leftDegree > -5:
            if leftDegree > 30:
                status = "error"
                result = "rc 0 0 0 0"
                return image, result, status
            if leftDegree > -5:
                status = "right"
                result = "rc 10 0 0 0"
            else:
                status = "notrealright"
                result = "rc -10 0 0 0"
        else:
            status  = "right"
            result = "rc 10 0 0 0"
        return image, result, status
    
    elif status == "leftright":
        result = "rc 0 22 0 0"
        return image, result, status
    
    if abs(left_vtx[1][0] - right_vtx[1][0]) <= 4:
        print("error here", left_vtx, right_vtx, left_vtx[0][0] - right_vtx[0][0])
        cv2.putText(image, "error", (left_vtx[0][0], left_vtx[0][1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA, False)
        status = "error"
        result = "rc 0 0 0 0"
        return image, result, status

    if leftDegree < -80 or rightDegree > 80:
        print("error")
        cv2.putText(image, "error", (left_vtx[0][0], left_vtx[0][1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA, False)
        status = "error"
        result = "rc 0 0 0 0"
        return image, result, status
    
    print('slope', left_slope+right_slope)
    lr_slope= int(left_slope+right_slope)
    
    # print(left_length, right_length)
    leftMainasRight = leftDegree - rightDegree
    leftMainasRight = int(leftMainasRight)
    # print(leftMainasRight, leftDegree, rightDegree)
    line_center = int((left_vtx[1][0] + right_vtx[1][0]) / 2)
    line_center_top = int((left_vtx[0][0] + right_vtx[0][0]) / 2)
    center = int(image.shape[1] / 2)
    top = int(image.shape[0] * 0.6)
    realTop = image.shape[0] - top
    left_bottom = line_center - left_vtx[1][0]
    left_top = line_center - left_fun(top)
    right_bottom = right_vtx[1][0] - line_center
    right_top = right_fun(top) - line_center
    left_total = ((left_bottom + left_top) * realTop) / 2
    right_total = ((right_bottom + right_top) * realTop) / 2
    mean= (left_total+right_total) / 2
    left_mean = left_total/mean*100
    right_mean = right_total/mean*100
    distanceToCenter = line_center - center
    # cv2.line(image, (line_center, 0), (line_center_top, image.shape[1]), (0, 0, 255), 5)
    cv2.line(image, (line_center, image.shape[1]), (line_center_top, 100), (0, 0, 255), 5)
    centerDegree = int(math.atan2(line_center_top-line_center, 100) * 180 / 3.14)
    try:
        center_slope = (100 - image.shape[1])/(line_center_top-line_center)
    except:
        center_slope = 100 - image.shape[1]
    print("center degree", centerDegree)
    cv2.line(image, (center, 0), (center, image.shape[0]), (0, 0, 0), 5)

    if lr_slope <= 1 and lr_slope >= -1:
        pass
    elif center < line_center:
        # distanceToCenter = abs(distanceToCenter)
        if distanceToCenter >= 7:
            status = "lefttoomuch "+str(distanceToCenter)
            result = "rc 10 0 0 0"
            return image, result, status
    elif center > line_center:
        # distanceToCenter = abs(distanceToCenter)
        if distanceToCenter <= 7:
            status = "righttoomuch " +str(distanceToCenter) 
            result = "rc -10 0 0 0"
            return image, result, status
    else:
        if distanceToCenter < 30:
            result = "rc 10 0 0 0"
            return image, result, status
        elif distanceToCenter < -30:
            result = "rc -10 0 0 0"
            return image, result, status
            
    print("centered")
    # print(center, line_center)

    if left_mean > right_mean:
        _centerDegree = abs(centerDegree)
        dx = left_mean-right_mean
        print(centerDegree)
        if dx > 20:
            if distanceToCenter <= 0 or centerDegree <= 0:
                result = "cw -3"
            else:
                result = "cw 3"
        else:
            if distanceToCenter < -15:
                result = "cw -3"
            elif distanceToCenter > 15:
                result = "cw 3"
            else:
                result = "cw 0"
            # print("cw 0")
            

    else:
        _centerDegree = abs(centerDegree)

        dx = right_mean-left_mean

        if dx > 20:
            if distanceToCenter <= 0:
                result = "cw -3"
            else:
                result = "cw 3"
             
        # elif dx > 15 and mode == "go":
        #     result = "cw "+str(centerDegree)
        #     print(result)
        else:
            if distanceToCenter < -15:
                result = "cw -3"
            elif distanceToCenter > 15:
                result = "cw 3"
            else:
                result = "cw 0"
    
    return image, result, status
# im = cv2.imread("test5.png")

# im, result, status = imageDegreeCheck(im, False)
# print(result, status)
# cv2.startWindowThread()
# cv2.namedWindow("ad", 0)
# cv2.resizeWindow("ad", 640, 480)
# cv2.imshow("ad", im)
# cv2.waitKey(0)
#TODO: how to calc the total? 

# cap = cv2.VideoCapture('outputqw.avi')
# lx = 0
# rx = 0
# while(cap.isOpened()):
#     cap.set(cv2.CAP_PROP_FPS, 10)
#     ret, frame = cap.read()
#     im, result,status = imageDegreeCheck(frame, 'go')
    
#     print(result, status)
#     cv2.imshow('frame',im)
#     key = cv2.waitKey(10)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#     if key == ord('p'):
#         cv2.waitKey(-1)
# cap.release()
# cv2.destroyAllWindows()