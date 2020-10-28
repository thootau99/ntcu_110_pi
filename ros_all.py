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
import datetime
import threading


from std_msgs.msg import String
import sensor_msgs.msg as msg
from rclpy.node import Node
from threading import Thread
from tello_msgs.srv import TelloAction
from cv_bridge import CvBridge

import detect_mask_video
from fix import check
from aruco import aru
from degree import imageDegreeCheck
from pprint import pprint
print('aaa')

# Initialize some variables

# Create arrays of known face encodings and their name

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber') # 初始化為 ROS-Node
        self.subscription = self.create_subscription(
            msg.Image,
            'image_raw',
            self.listener_callback,
            10)
            # 加入 ROS-subscrciber, 訂閱 "image" 取得影像
        self.subscriptionServer = self.create_subscription(
            String,
            'facenameset',
            self.userCommandCallBack,
            10)
        self.publisher_ = self.create_publisher(msg.Image, 'cvImage', 10) # 加入 ROS-publisher, 發出處理過的image
        self.publisher_facename = self.create_publisher(String, 'facename', 10)
        
        self.subscription  # prevent unused variable warning
        self.telloCli = self.create_client(TelloAction, 'tello_action') #TODO: 待完成，call ros2 service
        self.telloCliRequest = TelloAction.Request()
        if self.telloCli.wait_for_service:
            self.sendRequest("rc 0 0 0 0")
        self.bridge = CvBridge() # CvBridge Init
        self.process_this_frame = True # ! 用來一次只處理一個 frame 的 variable
        self.names = []  # read_file 存名字 
        self.labels = [] # read_file 存路徑
        self.noFaceCount = 0
        self.nameTest = 0
        self.noRepeatName = False
        self.noRepeatNameRecord = []
        self.L0 = 120   # 人與camera的距離
        self.S0 = 25600 # 預計的人臉框大小
        self.CX = 480   # 大約在畫面中間的 X 座標
        self.CY = 360   # 大約在畫面中間的 Y 座標
        self.xulyframe = 0
        self.future = 0
        self.action_future = False
        self.face_locations = [] # 存解析輸入圖片後人臉的位置
        self.face_encodings = [] # 存解析輸入圖片後人臉的 code
        self.face_names = []     # 存解析輸入圖片後人臉的 name
        self.stop = False
        self.face_location_record = [] # 記錄上次的人臉位置
        self.face_name_record = [] # 記錄上次的人臉名
        self.previous_have_name = False # 記錄上一次有沒有人名
        self.unknowTakeAgain = False # 
        self.unknownTakeAgainName = ''
        self.unknownTakeAgainCount = 0
        self.executeArcode = False
        self.known_face_encodings = [] # 存解析 database 中的人臉
        self.known_face_names = [] # 存解析 database 中的人名
        self.aruLock = False
        self.aruAlignSwitch = False
        self.aruBound = [100, 130]
        self.dark = False # 看圖有沒有過黑
        self.read_path('./dataset_img') # 到 /dataset_img 讀資料
        self.followName = 'uahuynh'        
        self.aruTarget = 0
        self.aruSide = []
        self.aruFar = [0,2,3,4,5,6,7,8]
        self.aruTempTarget = 0
        self.aruAligning = False
        self.aruGoMode = 'go'
        self.aruLocation = {'id': '', 'left': False, 'right': False, 'top': False, 'bottom': False}
        self.locationFixed = False
        self.aruLockCode = 'None'
        self.alignLock = False
        self.traceLocationAruco = False
        self.scannedArucoCode = []
        
        self.mode = 'mask'
        l = locals()
                    # if abs(aru_deg[index][0]) > 150 and abs(aru_deg[index][0]) < 168:
                    #     aru_deg[index][0] = abs(aru_deg[index][0])
                    #     print(aru_deg[index][0], "deg absed")

                    
                    # if aru_deg[index][0] < -90 and aru_deg[index][0] > -150:
                    #     instruction = ['rc', '0', '0', '0', '0']
                    #     instruction[4] = "-10"
                    #     instruction[2] = "-3"
                    #     print(aru_deg[index][0], "deg qua left")

                    # elif aru_deg[index][0] > 90 and aru_deg[index][0] < 150:
                    #     instruction = ['rc', '0', '0', '0', '0']
                    #     instruction[4] = '10'
                    #     instruction[2] = '-3'
                    #     print(aru_deg[index][0], "deg qua right")

        for item in self.names: # ! 讀取在 dataset_img 下的全部圖片
            l['%s_image'%item] = face_recognition.load_image_file("dataset_img/%s.jpg"%item)
            if len(face_recognition.face_encodings(l['%s_image'%item])) != 0:
                l['%s_face_encoding'%item] = face_recognition.face_encodings(l['%s_image'%item])[0]
                self.known_face_encodings.append(l['%s_face_encoding'%item])
                self.known_face_names.append(item)
            else:
                os.remove("./dataset_img/%s.jpg"%item)
                print("removed dataset_img/%s.jpg"%item)

        self.future = self.sendRequest('cw 20')

    def userCommandCallBack(self, s):
        print(s.data)
        if s.data == 'stop':
            self.stop = True 
            print("system stop", self.stop)
        elif s.data == 'start':
            self.stop = False
            print("system stop", self.stop)
        elif s.data == 'mask':
            self.mode = 'mask'
            print('system switch to mask mode')
        elif s.data == 'normal':
            self.mode = 'normal'
            print('system switch to normal mode')
        else:
            self.followName = s.data
            print(self.followName)

    def sendRequest(self, s):
        if type(s) != str:
            s = ' '.join(s)
        self.telloCliRequest.cmd = s
        future = self.telloCli.call_async(self.telloCliRequest)
        return future



    def setFollowName(self, s):
        self.followName = s

    def getFollowName(self):
        self.followName = self.nameTest + 1
        return self.followName

    def getFaceNames(self):
        s = String()
        if len(self.face_names) != 0:
            s.data = ' '.join(self.face_names)
        return s
    
    def aruBeLock(self, mode, s):
        self.aruLock = s
        self.aruLockCode = mode
    def listener_callback(self, image): #! 從image讀到cam image後觸發的 function
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8") # 轉換
        except CvBridgeError as e:
            print(e)
        small_frame = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5) # resize frame

        # gray_img = cv2.cvtColor(small_frame,cv2.COLOR_BGR2GRAY); # 轉灰階來辨別圖片有沒有過黑
        
        # r,c = gray_img.shape[:2] 
        # darkSum = 0
        # darkProp = 0
        # pixelSum = r*c

        # for row in gray_img:
        #     for col in row:
        #         if col < 40:
        #             darkSum += 1
        # darkProp = darkSum / pixelSum
        # if darkProp >= 0.75:
        #     rgb_small_frame = self.log(42, cv_image) # 若太黑就加亮
        #     rgb_small_frame = rgb_small_frame[:, :, ::-1] # 轉換成 face_recognition 的格式
        # else:
        #     rgb_small_frame = small_frame[:, :, ::-1] # 轉換成 face_recognition 的格式
        
        rgb_small_frame = small_frame[:, :, ::-1] # 轉換成 face_recognition 的格式
        # check(cv_image)
        aru_x = []
        aru_y = []
        aru_z = []
        aru_distance = []
        aru_id = []
        aru_deg = []
        align = False #TODO: test
        index = 0
        aru_x,aru_y,aru_z,aru_distance,aru_deg, aru_id,small_frame = aru(small_frame)
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", small_frame)
        cv2.waitKey(1)
        self.aruAlignSwitch = False
        if not self.future.done():
            return
        if self.action_future:
            small_frame, result_edge = imageDegreeCheck(small_frame, True)
            print(result_edge)
            self.sendRequest(result_edge)
            if self.aruTarget in self.aruFar:
                print("far target")
                self.aruBeLock("forward", True)
            self.action_future = False

        if self.aruLock:
            if self.aruLockCode == 'back':
                self.sendRequest('rc 0 -10 0 0')
            elif self.aruLockCode == 'forward':
                    self.sendRequest('rc 0 22 0 0')
        # if self.aruTarget == 3 and len(aru_id) == 0:
        #     self.sendRequest('rc -5 0 0 0')
        
        # if self.aruLocation['id'] == self.aruTarget and len(aru_id) == 0 and self.locationFixed:

        #     instruction = ['rc', '0', '-10', '0', '0']

        #     if self.aruLocation['left']:
        #         instruction[1] = '-7'
        #     elif self.aruLocation['right']:
        #         instruction[1] = '7'
             
        #     if self.aruLocation['top']:
        #         instruction[3] = '-7'
        #     elif self.aruLocation['bottom']:
        #         instruction[3] = '7'

        #     ins = ' '.join(instruction)
        #     self.sendRequest(instruction)
        if len(aru_id) != 0:
            if self.alignLock:
                align = True
            
            align = False
            x = False
            y = False
            print("Target:", self.aruTarget, aru_distance, aru_id)
            try:
                aru_distance[index] = aru_distance[index] * 5
            except:
                pass
            for ids in aru_id:
                side = False
                print(aru_id)
                if ids[0] in self.aruSide:
                    side = True
                if self.aruTarget == 3 and (ids[0] == 9 or ids[0] == 10):
                    if ids[0] == 9: # right
                        if aru_distance[index] < 140:
                            self.sendRequest("cw -10")
                    elif ids[0] == 10: #left
                        if aru_distance[index] < 140:
                            self.sendRequest("cw 10")
                if ids[0] == 14 and len(aru_id) == 1:
                    if not self.aruAlignSwitch:
                        self.aruBeLock('forward', False)
                    if self.aruTarget == 14 :
                        pass
                    else:
                        self.aruTempTarget = self.aruTarget
                    self.aruTarget = 14
                    self.aruAligning = True
                elif self.aruAligning == True:
                    _aru_id = np.array(aru_id).flatten()
                    print(_aru_id,self.aruTempTarget)
                    if self.aruTempTarget in _aru_id:
                        print(self.aruTempTarget)
                        self.aruTarget = self.aruTempTarget
                        self.aruAligning = False
                if ids[0] != self.aruTarget:
                    print("not turn")
                    continue
                self.locationFixed = False
                if ids[0] == self.aruTarget:
                    self.aruLocation['id' ] = ids[0]
                    self.locationFixed = True
                if ids[0] in self.aruFar and self.aruTarget == ids[0]:
                    if aru_distance[index] < 200:
                        self.aruBeLock('', False)
                elif ids[0] not in self.aruFar:
                    self.aruBeLock('', False)
                if ids[0] == 2 and self.aruTarget == 2:
                    self.aruBound = [130,400]
                elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'go':
                    self.aruBound = [120,130]
                elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'back':
                    self.aruBound = [120,150]
                else:
                    self.aruBound = [100, 200]

                print(self.aruLock, self.aruLockCode)
                # if ids[0] == 0 and self.aruTarget == 0:
                #     print("id[0] processing...", self.aruLock)
                #     aru_distance[index] = aru_distance[index] * 1.5
                #     if aru_distance[index] < 240:
                #         self.aruLock = False
                #         self.aruTarget = 5
                #         self.sendRequest("cw 90")
                #         self.alignLock = Falself.aruBoundse
                #         break
                # if ids[0] != self.aruTarget or self.aruLock:
                #     continue
                if ids[0] != self.aruTarget:
                    continue
                instruction = ['rc', '0', '0', '0', '0']
                # print(aru_distance[index])

                # print("X:",aru_x[index])
                # print("Y:", aru_y[index])
                if aru_distance[index] < self.aruBound[0]:
                    # self.sendRequest("rc 0 -10 0 0") # x z y raw
                    self.aruBeLock('back', True)
                    continue
                elif aru_distance[index] > self.aruBound[0] and self.aruLock == True:
                    self.aruBeLock('', False)
                if aru_distance[index] > self.aruBound[1]:
                    if aru_x[index] < -0.06 and not side:
                        instruction[1] = '-18'
                        instruction[2] = '0'
                        if self.locationFixed:
                            self.aruLocation['left'] = True
                            self.aruLocation['right'] = False
                    elif aru_x[index] > 10:
                        pass
                    elif aru_x[index] > 0.4 and not side:
                        print("qua left")
                        instruction[1] = '18'
                        instruction[2] = '0'
                        if ids[0] == 14:
                            instruction[1] = '25'
                        if self.locationFixed:
                            self.aruLocation['left'] = False
                            self.aruLocation['right'] = True
                    else:
                        if aru_distance[index] < 150:
                            instruction[1] = '0'
                            instruction[2] = '25'
                            if ids[0] == 14:
                                instruction[1] = '25'
                            if self.locationFixed:
                                self.aruLocation['left'] = False
                                self.aruLocation['right'] = False
                        else:
                            instruction[2] = '25'
                            
                        
                    print("y:",aru_y[index] < - 0.7, self.aruAligning)
                    if aru_y[index] < -0.7 and aru_distance[index] < 670 and not self.aruAligning:
                        print("up....")
                        instruction[3] = '18'
                        instruction[2] = '0'
                        if self.locationFixed:
                            self.aruLocation['bottom'] = True
                            self.aruLocation['top'] = False

                    elif aru_y[index] > 0.14 and aru_distance[index] < 670 and not self.aruAligning:
                        instruction[3] = '-10'
                        instruction[2] = '0'
                        if self.locationFixed:
                            self.aruLocation['bottom'] = False
                            self.aruLocation['top'] = True
                    else:
                        instruction[3] = '-5'
                        instruction[2] = '20'
                        if self.locationFixed:

                            self.aruLocation['bottom'] = False
                            self.aruLocation['top'] = False

                    # if abs(aru_deg[index][0]) > 150 and abs(aru_deg[index][0]) < 168:
                    #     aru_deg[index][0] = abs(aru_deg[index][0])
                    #     print(aru_deg[index][0], "deg absed")

                    
                            self.scannedArucoCode.append(0)
                    # if aru_deg[index][0] < -90 and aru_deg[index][0] > -150:
                    #     instruction = ['rc', '0', '0', '0', '0']
                    #     instruction[4] = "-10"
                    #     instruction[2] = "-3"
                    #     print(aru_deg[index][0], "deg qua left")

                    # elif aru_deg[index][0] > 90 and aru_deg[index][0] < 150:
                    #     instruction = ['rc', '0', '0', '0', '0']
                    #     instruction[4] = '10'
                    #     instruction[2] = '-3'
                    #     print(aru_deg[index][0], "deg qua right")

                    ins = ' '.join(instruction)
                    self.sendRequest(ins)
                elif not align:
                    print("z aligned")
                    self.aruBeLock('false', 'back')
                    if aru_x[index] < -0.06 and not side:
                        print("qua left")
                        instruction[1] = '-13'
                        if self.locationFixed:

                            self.aruLocation['left'] = True
                            self.aruLocation['right'] = False
                    elif aru_x[index] > 0.3 and not side:
                        print("qua right")
                        instruction[1] = '13'
                        if self.locationFixed:

                            self.aruLocation['left'] = False
                            self.aruLocation['right'] = True
                    else:
                        print("x aligned")
                        x = True
                        self.aruLocation['left'] = False
                        self.aruLocation['right'] = False

                    if aru_y[index] < -0.33 and not self.aruAligning:
                        print("y low")

                        instruction[3] = '13'
                        if self.locationFixed:
                            self.aruLocation['bottom'] = True
                            self.aruLocation['top'] = False
                    elif aru_y[index] > -0.01 and not self.aruAligning:
                        print("y high")
                        
                        instruction[3] = '-13'
                        if self.locationFixed:

                            self.aruLocation['bottom'] = False
                            self.aruLocation['top'] = True
                    else:
                        print("y aligned")
                        y = True
                        if self.locationFixed:
                        
                            self.aruLocation['bottom'] = False
                            self.aruLocation['top'] = False
                    if x and y:
                        align = True 
                    else:
                        ins = ' '.join(instruction)
                        self.sendRequest(ins)
                # print(ids[0], aru_distance[index])
                if align:
                    if ids[0] == 0 and self.aruTarget == 0 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 1
                            self.aruTempTarget = 1
                            self.future = self.sendRequest("cw -50")
                            self.action_future = True
                            self.aruGoMode = 'go'

                        

                            

                    elif ids[0] == 1 and self.aruTarget == 1 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 2
                            self.aruTempTarget = 2
                            self.future = self.sendRequest("cw 50")
                            self.action_future = True

                            self.aruGoMode = 'go'

                    
                    elif ids[0] == 2 and self.aruTarget == 2 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            # self.sendRequest("cw 30")

                            self.aruTarget = 3
                            self.aruTempTarget = 3
                            self.future = self.sendRequest("cw 40")
                            self.action_future = True



                            self.aruGoMode = 'go'
                    elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 4
                            self.aruTempTarget = 4
                            self.future = self.sendRequest("cw -90")
                            self.action_future = True



                    elif ids[0] == 4 and self.aruTarget == 4 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 3
                            self.aruTempTarget = 3
                            self.aruGoMode = 'back'
                            self.future = self.sendRequest("cw 180")
                            self.action_future = True


                    elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 5
                            self.aruTempTarget = 5
                            self.future = self.sendRequest("cw 90")
                            self.action_future = True


                    elif ids[0] == 5 and self.aruTarget == 5 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 6
                            self.aruTempTarget = 6
                            self.future = self.sendRequest("cw -50")
                            self.action_future = True



                    elif ids[0] == 6 and self.aruTarget == 6 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 7
                            self.aruTempTarget = 7
                            self.future = self.sendRequest("cw -30")
                            self.action_future = True



                    elif ids[0] == 7 and self.aruTarget == 7 and self.aruGoMode == 'back':
                        if aru_distance[index] < (self.aruBound[1] + 20):
                            self.aruTarget = 8
                            self.aruTempTarget = 8
                            self.future = self.sendRequest("cw 40")
                            self.action_future = True


                    elif ids[0] == 8 and self.aruTarget == 8 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.sendRequest("land")
                            self.aruTarget = 8
                            self.aruTempTarget = 8
                    elif ids[0] == 14 and self.aruTarget == 14:
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruAlignSwitch = True
                            self.aruBeLock('forward', True)
                    x = False
                    y = False
                
                index = index + 1
        elif self.aruTarget in self.aruFar and not self.aruLock:
            self.aruBeLock("forward", True)
        elif self.aruAligning == True:
            self.aruBeLock("forward", True)
        if self.stop == False and self.mode == 'normal':
            if self.process_this_frame:
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
                    for num in range(0,1,1):
                        if self.dark == True:
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
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
                        if name.split('_')[0] == self.followName:
                            if self.noRepeatName:
                                name = "Unknown"

                            self.noRepeatName = True

                        self.previous_have_name = True
                    if name == "Unknown" and not self.noRepeatName:
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
            if self.face_names == [] :
                self.noFaceCount += 1
                if self.noFaceCount > 20:
                    self.sendRequest('rc 0 0 0 20')
            else:
                self.noFaceCount = 0
            self.noRepeatName = False
            self.publisher_facename.publish(self.getFaceNames())
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 2 #y
                right *= 2 #x+w
                bottom *= 2 #y+h
                left *= 2 #x


                namePut = name.split('_')
                if self.unknowTakeAgain and name == self.unknownTakeAgainName and self.unknownTakeAgainCount > 3:
                    i = cv_image[top-50:bottom+50, left-50:right+50]
                    randstr = self.randomString(8)
                    cv2.imwrite('./dataset_img/%s_%s.jpg'%(self.unknownTakeAgainName, randstr),i)
                    self.unknowTakeAgain = False
                    self.unknownTakeAgainName = ''
                    self.unknownTakeAgainCount = 0
                execute = ['rc', '0', '0', '0', '0']
                if namePut[0] == self.followName:
                    width = right - left
                    height = bottom - top
                    dx = left + width/2 - self.CX ## x + w/2 - cx
                    dy = top + height/2 - self.CY ## y + h/2 - cy
                    d = round(self.L0 * m.sqrt(self.S0 / (width * height)))
                    xalign = False
                    yalign = False                                     
                    distanceAlign = False
                    
                    if self.xulyframe >= 1:
                        if dx > 130:
                            execute[4] = '13'
                        elif dx < -84:
                            execute[4] = '-13'
                        else:
                            execute[4] = '0'
                            xalign = True
                        if dy > 75:
                            execute[3] = '-13'
                        elif dy < -75:
                            execute[3] = '13'
                        else:
                            execute[3] = '0'
                            yalign = True
                        if (d-self.L0) > 15:
                            execute[2] = '13'

                        elif (d-self.L0) < -15:
                            execute[2] = '-13'
                        elif (d-self.L0) < -50:
                            self.send_request('emergency')
                            self.destroy_node()
                            rclpy.shutdown()
                            exit()
                        else:
                            execute[2] = '0'
                            distanceAlign = True
                        if xalign and yalign and distanceAlign:
                            cv2.putText(cv_image, "aligned", (left + 20, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

                        executeString = ' '.join(execute)
                        self.sendRequest(executeString)
                        self.xulyframe = 0
                    self.xulyframe += 1
                    # Draw a box around the face
                cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)
                


                    # Draw a label with a name below the face
                cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(cv_image, namePut[0], (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        elif self.stop == False and self.mode == "mask":
            (locs, preds) = detect_mask_video.main(small_frame)
            for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                startX = startX * 2
                startY = startY * 2
                endX = endX * 2
                endY = endY * 2
                (mask, withoutMask) = pred
                img = cv_image[startY:startY+endY, startX:startX+endX]

                faceLocation = face_recognition.face_locations(img)
                encoding = face_recognition.face_encodings(img ,faceLocation)
                
                # detect face name on frame?
                name = ""

                # for face_encoding in encoding:
                #     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    

                #     face_distance = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                #     best_match_index = np.argmin(face_distance)
                #     if matches[best_match_index]:
                #         name = self.known_face_names[best_match_index]
                #     name = name.split('_')
                #     print(name)
                #     name = name[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = name + "Mask" if mask > withoutMask else name +"No Mask"
                color = (0, 255, 0) if label == name + "Mask" else (0, 0, 255)
                # include the probability in the label

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(cv_image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(cv_image, (startX, startY), (endX, endY), color, 2)

        else:
            self.sendRequest('rc 0 0 0 0')
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
