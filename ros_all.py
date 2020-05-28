import cv2
import rclpy
import numpy as np
import json
import math as m
import os
from collections import Counter
import random
import string
import face_recognition

import sensor_msgs.msg as msg
from rclpy.node import Node

from tello_msgs.srv import TelloAction
from cv_bridge import CvBridge




# Initialize some variables

# Create arrays of known face encodings and their names



class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber') # 初始化為 ROS-Node
        self.subscription = self.create_subscription(
            msg.Image,
            'tello_ros/image_raw',
            self.listener_callback,
            10)
            # 加入 ROS-subscrciber, 訂閱 "image" 取得影像
        self.publisher_ = self.create_publisher(msg.Image, 'cvImage', 10) # 加入 ROS-publisher, 發出處理過的image
        self.subscription  # prevent unused variable warning
        self.telloCli = self.create_client(TelloAction, 'tello_action') #TODO: 待完成，call ros2 service
        while not self.telloCli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.telloCliRequest = TelloAction.Request()
        if self.telloCli.wait_for_service:
            self.sendRequest("rc 0 0 0 123")
        self.bridge = CvBridge() # CvBridge Init
        self.process_this_frame = True # ! 用來一次只處理一個 frame 的 variable
        self.names = []  # read_file 存名字 
        self.labels = [] # read_file 存路徑


        self.L0 = 120   # 人與camera的距離
        self.S0 = 25600 # 預計的人臉框大小
        self.CX = 480   # 大約在畫面中間的 X 座標
        self.CY = 360   # 大約在畫面中間的 Y 座標
    
        self.face_locations = [] # 存解析輸入圖片後人臉的位置
        self.face_encodings = [] # 存解析輸入圖片後人臉的 code
        self.face_names = []     # 存解析輸入圖片後人臉的 name

        self.face_location_record = [] # 記錄上次的人臉位置
        self.face_name_record = [] # 記錄上次的人臉名
        self.previous_have_name = False # 記錄上一次有沒有人名
        self.unknowTakeAgain = False # 
        self.unknownTakeAgainName = ''
        self.unknownTakeAgainCount = 0

        self.known_face_encodings = [] # 存解析 database 中的人臉
        self.known_face_names = [] # 存解析 database 中的人名

        self.dark = False # 看圖有沒有過黑
        self.read_path('./dataset_img') # 到 /dataset_img 讀資料

        l = locals()
        for item in self.names: # ! 讀取在 dataset_img 下的全部圖片
            l['%s_image'%item] = face_recognition.load_image_file("dataset_img/%s.jpg"%item)
            if len(face_recognition.face_encodings(l['%s_image'%item])) != 0:
                l['%s_face_encoding'%item] = face_recognition.face_encodings(l['%s_image'%item])[0]
                self.known_face_encodings.append(l['%s_face_encoding'%item])
                self.known_face_names.append(item)
            else:
                os.remove("./dataset_img/%s.jpg"%item)
                print("removed dataset_img/%s.jpg"%item)
    def sendRequest(self, s):
        self.telloCliRequest.cmd = s
        self.future = self.telloCli.call_async(self.telloCliRequest)
    def listener_callback(self, image): #! 從image讀到cam image後觸發的 function
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8") # 轉換
        except CvBridgeError as e:
            print(e)
        small_frame = cv2.resize(cv_image, (0, 0), fx=0.25, fy=0.25) # resize frame

        gray_img = cv2.cvtColor(small_frame,cv2.COLOR_BGR2GRAY); # 轉灰階來辨別圖片有沒有過黑
        r,c = gray_img.shape[:2] 
        darkSum = 0
        darkProp = 0
        pixelSum = r*c

        for row in gray_img:
            for col in row:
                if col < 40:
                    darkSum += 1
        darkProp = darkSum / pixelSum
        if darkProp >= 0.75:
            rgb_small_frame = self.log(42, cv_image) # 若太黑就加亮
            rgb_small_frame = rgb_small_frame[:, :, ::-1] # 轉換成 face_recognition 的格式
        else:
            rgb_small_frame = small_frame[:, :, ::-1] # 轉換成 face_recognition 的格式
        if self.process_this_frame:

            if self.dark:
                self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=3)
                # 若太黑就多掃描2次
                
            else:
                self.face_locations = face_recognition.face_locations(rgb_small_frame) # 掃描在圖片中的臉一次

            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            # 解析在畫面中的臉
            self.face_names = []
            if self.face_locations != '' and len(self.face_locations) != 0:
                self.face_location_record.append(self.face_locations)
            for face_encoding in self.face_encodings:
                matches = []
                matchesNamesCheckAgain = []

                name = "Unknown"
                for num in range(0,2,1):
                    if self.dark == True:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.8)
                    else:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                    face_dis = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    face_best = np.argmin(face_dis)
                    matchesNamesCheckAgain.append(face_best)
                self.dark = False
                result = Counter(matchesNamesCheckAgain)
                most_common = result.most_common()
                if matches[most_common[0][0]]:
                    name = self.known_face_names[most_common[0][0]]
                    self.face_name_record.append(most_common[0][0])
                    self.previous_have_name = True
                if name == "Unknown" :
                    dtop = 0
                    dright = 0
                    dbottom = 0
                    dleft = 0
                    if self.previous_have_name and len(self.face_location_record) > 2:
                        previous_top = self.face_location_record[-2][0][0]

                        previous_right = self.face_location_record[-2][0][1]

                        previous_botoom = self.face_location_record[-2][0][2]
                        previous_left = self.face_location_record[-2][0][3]

                        current_top = self.face_location_record[-1][0][0]
                        current_right = self.face_location_record[-1][0][1]
                        current_bottom = self.face_location_record[-1][0][2]
                        current_left = self.face_location_record[-1][0][3]

                        dtop = abs(previous_top - current_top)
                        dright = abs(previous_right - current_right)
                        dbottom = abs(previous_botoom - current_bottom)
                        dleft = abs(previous_left - current_left)
                    if self.previous_have_name and len(self.face_name_record) > 1 and dtop < 20 and dright < 20 and dbottom < 20 and dleft < 20:
                        name = self.known_face_names[self.face_name_record[-2]]
                        self.unknownTakeAgainCount = self.unknownTakeAgainCount + 1
                        self.previous_have_name = False
                        self.unknowTakeAgain = True
                        self.unknownTakeAgainName = name
                self.face_names.append(name)
        if len(self.face_location_record) > 5:
                self.face_location_record = []
        if len(self.face_name_record) > 5:
                self.face_name_record = []
        self.process_this_frame = not self.process_this_frame
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4 #y
            right *= 4 #x+w
            bottom *= 4 #y+h
            left *= 4 #x


            namePut = name.split('_')
            if self.unknowTakeAgain and name == self.unknownTakeAgainName and self.unknownTakeAgainCount > 3:
                i = cv_image[top-50:bottom+50, left-50:right+50]
                randstr = self.randomString(8)
                cv2.imwrite('./dataset_img/%s_%s.jpg'%(self.unknownTakeAgainName, randstr),i)
                self.unknowTakeAgain = False
                self.unknownTakeAgainName = ''
                self.unknownTakeAgainCount = 0

            if namePut[0] == 'uahuynhh':
                width = right - left
                height = bottom - top
                print(top, height, left, width)
                dx = left + width/2 - self.CX ## x + w/2 - cx
                dy = top + height/2 - self.CY ## y + h/2 - cy
                d = round(self.L0 * m.sqrt(self.S0 / (width * height)))
                xalign = False
                yalign = False
                distanceAlign = False
                #TODO:寫入飛行指令
                if dx > 114:
                    self.send_request('rc 20 0 0 0')
                elif dx < -68:
                    self.send_request('rc -20 0 0 0')
                else:
                    xalign = True
                if dy > 50:
                    self.send_request('rc 0 -20 0 0')
                elif dy < -50:
                    self.send_request('rc 0 20 0 0')
                else:
                    yalign = True
                if (d-self.L0) > 15:
                    self.send_request('rc 0 0 20 0')

                elif (d-self.L0) < -15:
                    self.send_request('rc 0 0 -20 0')
                else:
                    distanceAlign = True
                print(dx, dy, d-self.L0)
                if xalign and yalign and distanceAlign:
                    cv2.putText(cv_image, "aligned", (left + 20, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

                # Draw a box around the face
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)


                # Draw a label with a name below the face
            cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(cv_image, namePut[0], (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        transback = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.publisher_.publish(transback)
    def randomString(self, stringLength=8):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    def read_path(self, path_name):

        for dir_item in os.listdir(path_name):
            full_path = os.path.abspath(os.path.join(path_name, dir_item))

            if os.path.isdir(full_path):
                read_path(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path)
                    d = dir_item.split('.')
                    self.names.append(d[0])
                    self.labels.append(full_path)
            print(full_path)

        return self.names, self.labels
    def log(self, c, img):
        output = c * np.log(1.0+img)
        output = np.uint8(output+0.5)
        output = cv2.GaussianBlur(output, (3, 3), 0)
        self.dark = True
        return output
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
