import numpy as nu 
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person name: " : 1}
with open("labels.picle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color =frame[y:y+h,x:x+w]

        _id, conf = recognizer.predict(roi_gray)
        if conf>=4 and conf<=85:
            # print(_id)
            # print("it reaches here")
            print(labels[_id])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[_id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y), font,1,color, stroke, cv2.LINE_AA)
            # print(conf)
        img_item = 'my_img_test1.png'
        cv2.imwrite(img_item,roi_color)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
        subitems = smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),color,stroke)


    cv2.imshow('frame',frame)
    if(cv2.waitKey(20) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
