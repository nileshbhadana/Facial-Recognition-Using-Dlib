#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 03:42:15 2019

@author: nilesh
"""
#importing libraries
import cv2,os,pickle
import face_recognition as fr

dir_name="/home/nilesh/Desktop/face_recognition/dataset_images"

#creating directory
try:
    os.mkdir(dir_name)
except:
    print()
    
cam=cv2.VideoCapture(0)
counter=0


while cam.isOpened():
    frame=cam.read()[1]
    
    #converting BGR frame to RGB frame
    rgb_frame=frame[:,:,::-1]
    
    #getting locations of faces present
    faces=fr.face_locations(rgb_frame)
    for (top,right,bottom,left) in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        #saving images 
        cv2.imwrite(dir_name+"/"+"image"+str(counter)+".jpg",frame)
        counter=counter+1
        print(counter)
    
    cv2.imshow("live",frame)
    
    #handler
    if cv2.waitKey(100) & 0xFF==ord('q'):
        break
    if cv2.waitKey(100) & counter==1:
        break

cam.release()
cv2.destroyAllWindows()


#creating blank lists
known_face_encodings_list=[]
ids=[]
id=0
known_names=[]
face_names=os.listdir(dir_name)
for face_name in face_names:
    image_name=dir_name+"/"+face_name
    print(image_name)
    
    #loading images using face_recognition library
    known_face= fr.load_image_file(image_name)
    print(known_face)
    
    #getting encodings of faces
    known_face_encoding=fr.face_encodings(known_face)[0]
    print(known_face_encoding)
    
    #appending encodings into list
    known_face_encodings_list.append(known_face_encoding)
    known_names.append("nilesh")
    ids.append(id)
    
print(known_face_encodings_list)
print(len(known_face_encodings_list))


#storing data in files using pickle
with open("encodings.txt",'wb') as file_data:
    pickle.dump(known_face_encodings_list,file_data)

with open("name.txt",'wb') as file_data:
    pickle.dump(known_names,file_data)
    