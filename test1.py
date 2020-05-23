import face_recognition
import cv2
from cv_bridge import CvBridge
import rclpy
import numpy as np
import json
import math as m
import os
from collections import Counter
import random
import string
import sensor_msgs.msg as msg
from rclpy.node import Node





# Initialize some variables

# Create arrays of known face encodings and their names



class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            msg.Image,
            'image',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(msg.Image, 'cvImage', 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.process_this_frame = True
        self.names = []
        self.labels = []
        self.L0 = 120
        self.S0 = 25600
        self.CX = 480
        self.CY = 360
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_location_record = []
        self.face_name_record = []

        self.previous_have_name = False
        self.process_this_frame = True

        self.unknownChup = 0
        self.isUnknown = 0
        self.unknowTakeAgain = False
        self.unknownTakeAgainName = ''
        self.unknownTakeAgainCount = 0
        self.Distance = 100
        self.known_face_encodings = []
        self.known_face_names = []

        self.read_path('./dataset_img')

        l = locals()
        for item in self.names:
            l['%s_image'%item] = face_recognition.load_image_file("dataset_img/%s.jpg"%item)
            if len(face_recognition.face_encodings(l['%s_image'%item])) != 0:
                l['%s_face_encoding'%item] = face_recognition.face_encodings(l['%s_image'%item])[0]
                self.known_face_encodings.append(l['%s_face_encoding'%item])
                self.known_face_names.append(item)
            else:
                os.remove("./dataset_img/%s.jpg"%item)
                print("removed dataset_img/%s.jpg"%item)

    def listener_callback(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        small_frame = cv2.resize(cv_image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)

            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            self.face_names = []
            if self.face_locations != '' and len(self.face_locations) != 0:
                self.face_location_record.append(self.face_locations)
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)


                matchesNamesCheckAgain = []

                name = "Unknown"

                for num in range(0,3,1):
                    face_match = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                    face_dis = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    face_best = np.argmin(face_dis)
                    matchesNamesCheckAgain.append(face_best)

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
                        print(self.face_location_record)
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

                dx = left + width/2 - self.CX ## x + w/2 - cx
                dy = top + height/2 - self.CY ## y + h/2 - cy
                d = round(self.L0 * m.sqrt(self.S0 / (width * height)))
                xalign = False
                yalign = False
                distanceAlign = False
                # print(dx)
                if dx > 240:
                    print("move right...")
                elif dx < 48:
                    print("move left...")
                else:
                    xalign = True
                if dy > 50:
                    print("move down...")
                elif dy < -50:
                    print("move up")
                else:
                    yalign = True
                if (d-self.L0) > 15:
                    print("move front...")
                elif (d-self.L0) < -15:
                    print("move back")
                else:
                    distanceAlign = True
                print(dx, dy, d-self.L0)
                if xalign and yalign and distanceAlign:
                    cv2.putText(cv_image, "aligned", (left + 20, bottom + 20), font, 1.0, (255, 255, 255), 1)

                # Draw a box around the face
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)


                # Draw a label with a name below the face
            cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(cv_image, namePut[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

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
