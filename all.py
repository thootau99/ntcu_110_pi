#coding=utf-8
import face_recognition
import cv2
import numpy as np
import json
import math as m
import os
from collections import Counter
import random
import string

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
names = []
labels = []
L0 = 120
S0 = 25600
CX = 480
CY = 360

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def read_path(path_name):

    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                d = dir_item.split('.')
                names.append(d[0])
                labels.append(full_path)
        print(full_path)

    return names, labels
read_path('./dataset_img')

known_face_encodings = []
known_face_names = []
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
#face_recognition.
l = locals()
for item in names:
    l['%s_image'%item] = face_recognition.load_image_file("dataset_img/%s.jpg"%item)
    if len(face_recognition.face_encodings(l['%s_image'%item])) != 0:
        l['%s_face_encoding'%item] = face_recognition.face_encodings(l['%s_image'%item])[0]
        known_face_encodings.append(l['%s_face_encoding'%item])
        known_face_names.append(item)
    else:
        os.remove("./dataset_img/%s.jpg"%item)
        print("removed dataset_img/%s.jpg"%item)
# Create arrays of known face encodings and their names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_location_record = []
face_name_record = []

previous_have_name = False
process_this_frame = True

unknownChup = 0
isUnknown = 0
unknowTakeAgain = False
unknownTakeAgainName = ''
unknownTakeAgainCount = 0
Distance = 100

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # 將影片大小降為1/4, 好做辨識
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # OpenCV-> BGR, face_recognition -> RGB (轉換)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        if face_locations != '' and len(face_locations) != 0:
            face_location_record.append(face_locations)
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            matchesNamesCheckAgain = []

            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            for num in range(0,3,1):
                face_match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_dis = face_recognition.face_distance(known_face_encodings, face_encoding)
                face_best = np.argmin(face_dis)
                matchesNamesCheckAgain.append(face_best)

            # Or instead, use the known face with the smallest distance to the new face

            result = Counter(matchesNamesCheckAgain)
            most_common = result.most_common()
            if matches[most_common[0][0]]:
                name = known_face_names[most_common[0][0]]
                face_name_record.append(most_common[0][0])
                previous_have_name = True
            if name == "Unknown" :
                if previous_have_name and len(face_location_record) > 2:
                    print(face_location_record)
                    previous_top = face_location_record[-2][0][0]

                    previous_right = face_location_record[-2][0][1]

                    previous_botoom = face_location_record[-2][0][2]
                    previous_left = face_location_record[-2][0][3]

                    current_top = face_location_record[-1][0][0]
                    current_right = face_location_record[-1][0][1]
                    current_bottom = face_location_record[-1][0][2]
                    current_left = face_location_record[-1][0][3]

                    dtop = abs(previous_top - current_top)
                    dright = abs(previous_right - current_right)
                    dbottom = abs(previous_botoom - current_bottom)
                    dleft = abs(previous_left - current_left)
                    print(previous_have_name, len(face_name_record))
                    if previous_have_name and len(face_name_record) > 1 and dtop < 20 and dright < 20 and dbottom < 20 and dleft < 20:
                        name = known_face_names[face_name_record[-2]]
                        unknownTakeAgainCount = unknownTakeAgainCount + 1
                        previous_have_name = False
                        unknowTakeAgain = True
                        unknownTakeAgainName = name
                isUnknown = isUnknown + 1
                if isUnknown > 20:
                    i = frame[top-10:bottom+10, left-10:right+10]

                    ask = input("是否要新增為信任使用者")
                    if ask == "y":
                        ten = input("請輸入名字")

                        cv2.imwrite('./dataset_img/%s.jpg'%ten,i)
                    isUnknown = 0

            face_names.append(name)
    if len(face_location_record) > 5:
        face_location_record = []
    if len(face_name_record) > 5:
        face_name_record = []
    process_this_frame = not process_this_frame

    #150-200 450-500 328 340 100cm, center

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4 #y
        right *= 4 #x+w
        bottom *= 4 #y+h
        left *= 4 #x
        namePut = name.split('_')
        print(video_capture.get(3), video_capture(4))
        if unknowTakeAgain and name == unknownTakeAgainName and unknownTakeAgainCount > 3:

            i = frame[top-50:bottom+50, left-50:right+50]
            randstr = randomString(8)
            cv2.imwrite('./dataset_img/%s_%s.jpg'%(unknownTakeAgainName, randstr),i)
            unknowTakeAgain = False
            unknownTakeAgainName = ''
            unknownTakeAgainCount = 0
        if namePut[0] == 'uahuynhh':
            width = right - left
            height = bottom - top
            print(left, top, width, height)
            dx = left + width/2 - CX ## x + w/2 - cx
            dy = top + height/2 - CY ## y + h/2 - cy
            d = round(L0 * m.sqrt(S0 / (width * height)))
            xalign = False
            yalign = False
            distanceAlign = False
            print(d-L0)
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

            if (d-L0) > 15:
                print("move front...")
            elif (d-L0) < -15:
                print("move back")
            else:
                distanceAlign = True
            if xalign and yalign and distanceAlign:
                cv2.putText(frame, "aligned", (left + 20, bottom + 20), font, 1.0, (255, 255, 255), 1)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        namePut[0] = namePut[0] + str(dx) + str(dy)
        cv2.putText(frame, namePut[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
