import os
import cv2
from PIL import Image
import numpy as np 
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

image_dir = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids = {}
x_train = []
y_label = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            _id = label_ids[label]
            # print(label_ids)
            # x_train.append(path)
            # y_label.append(label)
            pil_img =Image.open(path).convert("L")
            size = (550,550)
            final_img = pil_img.resize(size,Image.ANTIALIAS)
            image_arr = np.array(final_img,"uint8")
            # print(image_arr)
            faces = face_cascade.detectMultiScale(image_arr,scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_arr[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(_id)
# print(x_train)
# print(y_label)
with open("labels.picle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_label))
recognizer.save("trainner.yml")