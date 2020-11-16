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
import threading, queue
import time
import asyncio
from image_uploader.image_uploader import upload as u
from std_msgs.msg import String
import sensor_msgs.msg as msg
from rclpy.node import Node
from threading import Thread
from tello_msgs.msg import TelloResponse
from tello_msgs.srv import TelloAction
from tello_msgs.msg import FlightData
from cv_bridge import CvBridge

from fix import check
from aruco import aru
from degree_new_test import imageDegreeCheck
from pprint import pprint
import uuid
# print('aaa')
frame_queue = queue.Queue()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frame_save_count = 0
upload_lock = False
# Initialize some variables
def face_detect():
    global frame_queue
    global face_cascade
    if not frame_queue.empty():
        frame = frame_queue.get()
            # TODO: get something face detcet stuff into here.
                        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return
        else:
            for (x,y,w,h) in faces:
                face = frame[int(y):int(y+h), int(x):int(x+w)]
                crop = "crop/" + str(uuid.uuid4().hex) + ".jpg"
                cv2.imwrite(crop, face)
            
            fname = "save/" + str(uuid.uuid4().hex) + ".jpg"
            cv2.imwrite(fname, frame)
            upload(fname)

def upload(img_path):
    global upload_lock
    if not upload_lock:
        upload_lock = True
        u(img_path)
        upload_lock = False

# Create arrays of known face encodings and their name
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber') # 初始化為 ROS-Node
        self.subscription = self.create_subscription(
            msg.Image,
            'image_raw',
            self.listener_callback,
            10)

        self.responseFromTello = self.create_subscription(
            TelloResponse,
            'tello_response',
            self.responseCallBack,
            10)

        self.responseFromTello = self.create_subscription(
            FlightData,
            'flight_data',
            self.heightCallback,
            10)
            # 加入 ROS-subscrciber, 訂閱 "image" 取得影像
        self.subscriptionServer = self.create_subscription(
            String,
            'facenameset',
            self.userCommandCallBack,
            10)
        self.publisher_ = self.create_publisher(msg.Image, 'cvImage', 10) # 加入 ROS-publisher, 發出處理過的image
        self.publisher_facename = self.create_publisher(String, 'facename', 10)
        self.action_processing = False
        self.subscription  # prevent unused variable warning
        self.telloCli = self.create_client(TelloAction, 'tello_action') #TODO: 待完成，call ros2 service
        self.telloCliRequest = TelloAction.Request()
        self.cwQueue = queue.Queue()
        if self.telloCli.wait_for_service():
            self.sendRequest("rc 0 0 0 0")
        self.bridge = CvBridge() # CvBridge Init
        self.process_this_frame = True # ! 用來一次只處理一個 frame 的 variable
        self.names = []  # read_file 存名字 
        self.labels = [] # read_file 存路徑
        self.noFaceCount = 0
        self.yaw = 0
        self.inityaw = 0
        self.frame_count = 0
        self.realyaw = 0
        self.yawWrite = True
        self.nameTest = 0
        self.noRepeatName = False
        self.noRepeatNameRecord = []
        self.cwSwitch = False
        self.L0 = 120   # 人與camera的距離
        self.S0 = 25600 # 預計的人臉框大小
        self.CX = 480   # 大約在畫面中間的 X 座標
        self.CY = 360   # 大約在畫面中間的 Y 座標
        self.xulyframe = 0
        self.future = 0
        self.cwCount = 10
        self.successconb = False
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
        self.noLineStatus = ""
        self.noLineInstruction = ""
        self.heightLowLock = False
        self.heightHighLock = False
        self.xdistanceLock = 600
        self.landautien = True
        self.known_face_encodings = [] # 存解析 database 中的人臉
        self.known_face_names = [] # 存解析 database 中的人名
        self.aruLock = False
        self.keep = False
        self.batteryCount = 0
        self.battery = "100"
        self.aruAlignSwitch = False
        self.aruBound = [100, 130]
        self.dark = False # 看圖有沒有過黑
        self.read_path('./dataset_img') # 到 /dataset_img 讀資料
        self.followName = 'uahuynh'
        self.aruTarget = 2
        self.futureInstruction = ""
        self.aruSide = []
        self.aruFar = [0,2,3,4,5,6,7,8]
        self.xwide = [5, 6, 7, 8,9]
        self.aruTempTarget = 0
        self.aruAligning = False
        self.aruGoMode = 'go'
        self.aruLocation = {'id': '', 'left': False, 'right': False, 'top': False, 'bottom': False}
        self.locationFixed = False
        self.aruLockCode = 'None'
        self.alignLock = False
        self.traceLocationAruco = False
        self.scannedArucoCode = []
        self.combArray = []
        self.combError = 0
        self.mode = ''
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

        self.future = self.sendRequest('rc 0 0 0 0')
    def getDegreeActually(self):
        if self.aruGoMode == "go":
            base = 180
        else:
            base = 0
        return self.yaw - base
    def userCommandCallBack(self, s):
        # print(s.data)
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
    # def battery(self):
    #     if self.batteryFuture == False:
    #         self.batteryFuture = await self.sendRequest("battery?")
    #         return
    #     if ():
    #         print(f.result())
    def sendRequest(self, s, normal=True):
        if not self.action_processing:
            if type(s) != str:
                s = ' '.join(s)
            self.telloCliRequest.cmd = s
            future = self.telloCli.call_async(self.telloCliRequest)
            return future
        else:
            if not normal:
                if type(s) != str:
                    s = ' '.join(s)
                self.telloCliRequest.cmd = s
                future = self.telloCli.call_async(self.telloCliRequest)
                self.action_processing = False
                return future
            else:
                return
        
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
    def heightCallback(self, data):
        self.battery = str(data.bat)
        self.yaw = str(data.yaw)
        # print(data)
        if self.yawWrite:
            self.yaw = data.yaw

        if data.h == 0:
            self.heightLowLock = False
            return
        if data.h < 150:
            print("h low")
            self.heightLowLock = True
        elif data.h > 200:
            print("h high")
            self.heightHighLock = True
        else:
            self.heightLowLock = False
            self.heightHighLock = False

    def responseCallBack(self,data):
        print("CwQueue",self.cwQueue.queue)
        if 'error' in data.str:
            print(self.cwQueue.empty(), self.cwQueue.qsize())
            if self.cwQueue.empty():
                return
            try:
                instruction = self.cwQueue.queue[0]
            # if instruction == "cw 3" or instruction == "cw -3":
            #     print("30 yet",instruction)
            #     if self.cwCount > 10:
            #         return
                self.cwQueue.put(instruction)
                self.cwQueue.get()
                self.sendRequest(instruction)
            except:
                pass
            # self.cwSwitch = True
        else:
            if self.cwQueue.empty():
                # self.cwSwitch = False
                return
            else:
                # self.cwSwitch = False
                instruction = self.cwQueue.get()
                # if instruction == "battery?":
                #     self.battery = str(data.str)

    def listener_callback(self, image): #! 從image讀到cam image後觸發的 function
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8") # 轉換
        except CvBridgeError as e:
            print(e)
        small_frame = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5) # resize frame
        if self.frame_count % 30 == 0:
            frame_queue.put(small_frame)
            fd = threading.Thread(target=face_detect)
            fd.start()
        else:
            self.frame_count = self.frame_count + 1
        # print(self.future._result)
        if self.landautien:
            self.landautien = False
            # self.action_future = True
            self.yawWrite = True
            
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
        # if self.batteryCount > 30:
        #     self.sendRequest("battery?")
        #     self.batteryCount = 0
        # else:
        #     self.batteryCount = self.batteryCount + 1
        batteryStr = "battery: " + self.battery
        yawStr = "yaw: " + str(self.yaw)
        statusText = batteryStr + yawStr
        cv2.putText(small_frame, statusText, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", small_frame)
        cv2.waitKey(1)
        self.aruAlignSwitch = False
        
        if self.heightLowLock:
            self.sendRequest("rc 0 0 25 0")
            return
        
        if self.heightHighLock:
            self.sendRequest("rc 0 0 -10 0")
            return

        if not self.cwQueue.empty():
            # self.sendRequest("rc 0 0 0 0")
            return
        if self.futureInstruction != "":
            print(self.futureInstruction)
            self.cwQueue.put(self.futureInstruction)
            self.sendRequest(self.futureInstruction)
            self.futureInstruction = ""
            return
        _, _, distanceToCenter = imageDegreeCheck(cv_image, False)
        try:
            if "cw" in r:
                print("cw", r)
        except:
            pass
        if self.action_future:
            self.combError = self.combError + 1
            if self.combError > 20:
                self.action_future = False
                self.successconb = False
                self.combError = 0
                if self.aruTarget in self.aruFar:
                    print("far target")
                    self.aruBeLock("forward", True)
                return
            self.successconb = False
            if not self.successconb:
                # if self.aruGoMode == "go": 
                _, result_edge, status = imageDegreeCheck(cv_image, True)
                print(status, self.noLineStatus, self.noLineInstruction)
                if status != "":
                    if status == "lefttoomuch":
                        self.noLineStatus = ""
                        self.noLineInstruction = ""
                    if status == "righttoomuch":
                        self.noLineStatus = ""
                        self.noLineInstruction = ""
                    if status == 'notrealleft':
                        if self.noLineStatus == "left":
                            self.noLineStatus = 'notrealleft'
                            self.noLineInstruction = "rc 10 0 0 0"
                        self.noLineStatus = 'notrealleft'
                        self.noLineInstruction = "rc 10 0 0 0"
                    if status == 'left':
                        if self.noLineInstruction == 'notrealleft':
                            self.noLineStatus = 'notrealleft'
                            # self.noLineInstruction = "rc 10 0 0 0"

                        else:
                            self.noLineStatus = 'left'
                            # self.noLineInstruction = 'rc -10 0 0 0'
                        # return
                    if status == 'right':
                        if self.noLineStatus == "notrealright":
                            self.noLineStatus = "notrealright"
                            # self.noLineInstruction = "rc -10 0 0 0"
                        else:
                            self.noLineStatus = "right"
                            # self.noLineInstruction = "rc 10 0 0 0"
                        # return
                    if status == "notrealright":
                        # print(self.aruGoMode)
                        if self.aruGoMode == "go":
                            self.noLineStatus = "notrealleft"
                            self.noLineInstruction = "rc 10 0 0 0"
                        else:    
                            self.noLineStatus = "notrealright"
                            self.noLineInstruction = "rc -10 0 0 0"
                        # return
                    
                if self.noLineStatus != "":
                    
                    if status == "":
                        self.noLineStatus = ""
                        self.noLineInstruction = ""
                    else:
                        print("entered")
                        if self.noLineStatus == "left" or self.noLineStatus == "notrealright":
                            if self.aruGoMode == "go" and self.noLineStatus == "notrealright":
                                self.noLineStatus = "notrealleft"
                            self.noLineInstruction = "rc -10 0 0 0"
                        
                        if self.noLineStatus == "right" or self.noLineStatus == "notrealleft":
                            self.noLineInstruction = "rc 10 0 0 0"
                        self.sendRequest(self.noLineInstruction)
                        return
                if result_edge == 0:
                    print('conb failed')
                    return
                else:
                    self.successconb = True
                    print(status, result_edge)
                    if result_edge == "" or result_edge == "cw 0":
                        if self.cwSwitch:
                            return
                        self.keep = False
                        self.action_future = False
                        self.successconb = False
                        self.combError = 0
                        if self.aruTarget in self.aruFar:
                            print("far target")
                            self.aruBeLock("forward", True)
                        return
                    else:
                        if "cw" in result_edge:
                            print("self.cwCount", self.cwCount)
                            if self.cwCount < 10:
                                self.cwCount = self.cwCount + 1
                                print(self.cwCount)
                                self.sendRequest("rc 0 0 0 0")
                                self.action_future = False
                                return
                            if self.cwCount >= 10:
                                print("cwed")
                                self.sendRequest("rc 0 0 0 0", False)
                                self.cwCount = 0
                                # self.cwSwitch = True
                                self.cwQueue.put(result_edge)
                                self.future = self.sendRequest(result_edge, False)
                                return
                                
                        self.future = self.sendRequest(result_edge, False)
                        # self.keep = False
                        # self.action_future = False
                        # self.successconb = False
                        # if self.aruTarget in self.aruFar:
                        #     print("far target")
                        #     self.aruBeLock("forward", True)
                        return
                    # print(result_edge)
        if self.aruTarget in self.aruFar:
                    print("far target")
                    self.aruBeLock("forward", True)
        distanceToCenterFix = 0
        try:
            distanceToCenter = int(distanceToCenter)
            if distanceToCenter > 30:
                distanceToCenterFix = 10
            elif distanceToCenter < -30:
                distanceToCenterFix = -10
            # print(distanceToCenter, "distnaceToCenter")
        except Exception as e:
            # print(distanceToCenter, "distanceToCenter error")
            # print(e)
            pass
            

        if self.aruLock:
            if self.aruLockCode == 'back':
                self.sendRequest('rc 0 -10 0 0')
            elif self.aruLockCode == 'forward':
                if len(aru_id) == 0:
                    # print(distanceToCenter, "distnaceToCenter")
                    request = "rc "+str(distanceToCenterFix)+" 25 -3 0"
                    print(request, "request")
                    self.sendRequest(request)
                else:
                    # print(distanceToCenter, "distnaceToCenter")
                    request = "rc "+str(distanceToCenterFix)+" 22 -3 0"
                    print(request, "request")                    
                    self.sendRequest(request)

        if len(aru_id) != 0:
            _aru_id = np.array(aru_id).flatten()
            if self.alignLock:
                align = True
            align = False
            x = False
            y = False
            try:
                aru_distance[index] = aru_distance[index] * 5
            except:
                pass
            for ids in aru_id:

                
                side = False
                if ids[0] in self.aruSide:
                    side = True
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
                    # print(_aru_id,self.aruTempTarget)
                    if self.aruTempTarget in _aru_id:
                        # print(self.aruTempTarget)
                        self.aruTarget = self.aruTempTarget
                        self.aruAligning = False

                
                if ids[0] == 9 and not self.action_future:
                    if 9 in self.combArray:
                        pass
                    else:
                        self.action_future = True
                        self.combArray.append(ids[0])
                        print(self.combArray)
                        return

                if ids[0] == 10 and not self.action_future:
                    if 10 in self.combArray:
                        pass
                    else:
                        self.action_future = True
                        self.combArray.append(ids[0])
                        print(self.combArray)

                        return 

                if ids[0] != self.aruTarget:
                    print("not turn")
                    continue
        
                # print("Target:", self.aruTarget, aru_distance, aru_id, ids[0])
                


                self.locationFixed = False
                if ids[0] == self.aruTarget:
                    self.aruLocation['id' ] = ids[0]
                    self.locationFixed = True
                if ids[0] in self.aruFar and self.aruTarget == ids[0]:
                    # print(aru_distance, index)
                    if aru_distance[index] < 200:
                        self.aruBeLock('', False)
                elif ids[0] not in self.aruFar:
                    self.aruBeLock('', False)
                if ids[0] == 2 and self.aruTarget == 2:
                    self.aruBound = [130,400]
                elif ids[0] == 6 and self.aruTarget == 6:
                    self.aruBound = [200,300]
                elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'go':
                    self.aruBound = [120,130]
                elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'back':
                    self.aruBound = [120,150]
                else:
                    self.aruBound = [100, 200]

                if ids[0] in self.xwide and self.aruTarget in self.xwide:
                    self.xdistanceLock = 800
                else:
                    self.xdistanceLock = 600
                # print(self.aruLock, self.aruLockCode)
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
                try:
                    if aru_distance[index] < self.aruBound[0]:
                        # self.sendRequest("rc 0 -10 0 0") # x z y raw
                        self.aruBeLock('back', True)
                        continue
                    elif aru_distance[index] > self.aruBound[0] and self.aruLock == True:
                        self.aruBeLock('', False)
                except:
                    pirnt(aru_distance, self.aruBound)
                # print(aru_x[index])
                if aru_distance[index] > self.aruBound[1]:
                    if aru_x[index] < -0.06 and not side and len(aru_id) == 1 and  aru_distance[index] < self.xdistanceLock:
                        instruction[1] = '-18'
                        instruction[2] = '0'
                        if self.locationFixed:
                            self.aruLocation['left'] = True
                            self.aruLocation['right'] = False
                    elif aru_x[index] > 10:
                        pass
                    elif aru_x[index] > 0.4 and not side and aru_distance[index] < self.xdistanceLock and len(aru_id) == 1:
                        # print("qua left")
                        instruction[1] = '18'
                        instruction[2] = '0'
                        if ids[0] == 14 or ids[0] == 12:
                            instruction[1] = '25'
                        if self.locationFixed:
                            self.aruLocation['left'] = False
                            self.aruLocation['right'] = True
                    else:
                        if aru_distance[index] < 150:
                            instruction[1] = '0'
                            instruction[2] = '25'
                            if ids[0] == 14 :
                                instruction[1] = '25'
                            if self.locationFixed:
                                self.aruLocation['left'] = False
                                self.aruLocation['right'] = False
                        else:
                            instruction[2] = '25'
                            
                        
                    # print("y:",aru_y[index] < - 0.7, self.aruAligning)
                    if aru_y[index] < -0.7 and aru_distance[index] < 670 and not self.aruAligning:
                        # print("up....")
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
                        if instruction[1] == '0':
                            instruction[2] = '20'
                        if self.locationFixed:

                            self.aruLocation['bottom'] = False
                            self.aruLocation['top'] = False


                    
                            self.scannedArucoCode.append(0)
                 
                    ins = ' '.join(instruction)
                    self.sendRequest(ins)
                elif not align:
                    print("z aligned")
                    self.aruBeLock('false', 'back')
                    if aru_x[index] < -0.06 and not side:
                        instruction[1] = '-13'
                        if self.locationFixed:

                            self.aruLocation['left'] = True
                            self.aruLocation['right'] = False
                    elif aru_x[index] > 0.3 and not side:
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
                        instruction[3] = '-18'
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
                if align:
                    if ids[0] == 0 and self.aruTarget == 0 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 1
                            self.aruTempTarget = 1
                            self.action_processing = True
                            self.future = self.sendRequest("cw -50", False)
                            self.cwQueue.put("cw -50")
                            self.action_future = True
                            self.aruGoMode = 'go'

                        

                            

                    elif ids[0] == 1 and self.aruTarget == 1 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 2
                            self.aruTempTarget = 2
                            self.action_processing = True
                            self.future = self.sendRequest("cw 50", False)
                            self.cwQueue.put("cw 50")
                            self.aruBeLock("forward", True)

                            self.aruGoMode = 'go'

                    
                    elif ids[0] == 2 and self.aruTarget == 2 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            # self.sendRequest("cw 30")

                            self.aruTarget = 6
                            self.aruTempTarget = 6
                            self.action_processing = True
                            self.cwQueue.put("cw 40")
                            self.future = self.sendRequest("cw 40", False)
                            self.futureInstruction = "cw 140"
                            
                            self.action_future = True
                            self.aruBeLock("forward", True)
                            self.combArray.clear()
                            self.aruGoMode = 'back'
                    elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 4
                            self.aruTempTarget = 4
                            self.action_processing = True
                            self.cwQueue.put("cw -90")
                            self.sendRequest("cw -90", False)
                            # self.action_future = True
                            self.aruBeLock("forward", True)



                    elif ids[0] == 4 and self.aruTarget == 4 and self.aruGoMode == 'go':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 3
                            self.aruTempTarget = 3
                            self.aruGoMode = 'back'
                            self.action_processing = True
                            self.cwQueue.put("cw 180")

                            self.future = self.sendRequest("cw 180", False)
                            self.combArray.clear()
                            self.action_future = True


                    elif ids[0] == 3 and self.aruTarget == 3 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 5
                            self.aruTempTarget = 5
                            self.action_processing = True
                            self.cwQueue.put("cw 90")

                            self.future = self.sendRequest("cw 90", False)
                            self.action_future = True


                    elif ids[0] == 5 and self.aruTarget == 5 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 2
                            self.aruTempTarget = 2
                            self.cwQueue.put("cw 90")
                            self.sendRequest("cw 90", False)
                            self.aruBeLock("forward", True)

                    elif ids[0] == 2 and self.aruTarget == 2 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 2
                            self.aruTempTarget = 2
                            self.sendRequest("land", False)

                            self.aruGoMode = 'go'

                    elif ids[0] == 6 and self.aruTarget == 6 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.aruTarget = 2
                            self.aruTempTarget = 2
                            self.action_processing = True
                            self.cwQueue.put("cw -40")
                            self.future = self.sendRequest("cw -40", False)
                            self.futureInstruction = "cw -140"
                            
                            self.action_future = True
                            self.aruBeLock("forward", True)
                            self.combArray.clear()

                            self.aruGoMode = 'go'




                    elif ids[0] == 7 and self.aruTarget == 7 and self.aruGoMode == 'back':
                        if aru_distance[index] < (self.aruBound[1] + 20):
                            self.aruTarget = 8
                            self.aruTempTarget = 8
                            self.action_processing = True
                            self.cwQueue.put("cw 90")
                            self.future = self.sendRequest("cw 90", False)
                            # self.action_future = True


                    elif ids[0] == 8 and self.aruTarget == 8 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.cwQueue.put("cw -50")
                            self.sendRequest("cw -50")
                            self.aruTarget = 11
                            self.aruTempTarget = 11
                            self.aruBeLock("forward", True)

                    elif ids[0] == 11 and self.aruTarget == 11 and self.aruGoMode == 'back':
                        if aru_distance[index] < self.aruBound[1]:
                            self.sendRequest("land")
                            self.aruTarget = 11
                            self.aruTempTarget = 11
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
                            execute[3] = '-18'
                        elif dy < -75:
                            execute[3] = '18'
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
                import detect_mask_video 
                (locs, preds) = detect_mask_video.main(small_frame)
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
            pass
            # self.sendRequest('rc 0 0 0 0')
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
