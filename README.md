# drowsyness-detection
Project Overview: Drowsiness Detection System Using Python and Streamlit

What:
The Drowsiness Detection System is a real-time application designed to monitor and detect signs of drowsiness in individuals, particularly drivers, to enhance safety. The system uses computer vision techniques to analyze eye movements and facial features, identifying when the user's eyes are closed for an extended period, which is a common sign of drowsiness. When drowsiness is detected, the system triggers an alert to wake the user.

Key Features:
- Real-time video feed capture and processing.
- Eye aspect ratio (EAR) calculation for drowsiness detection.
- Visual and/or auditory alerts when drowsiness is detected.
- User-friendly interface implemented using Streamlit.

Why:
Drowsiness while driving or during critical tasks is a major safety concern, leading to accidents, injuries, and fatalities. A drowsiness detection system aims to prevent such incidents by providing early warnings to the user, allowing them to take necessary actions such as resting or taking a break. This system can significantly improve safety in various contexts, including driving, operating heavy machinery, and monitoring critical operations.

HOW:
1.	Backend:
a.	Connection with ip camera -> Open CV
b.	Training The Model
c.	Eye detection		-> Open CV
d.	Closed Eye Detection	-> Keras
e.	Savedata			-> Open CV
2.	Front END:
Web Application - > Streamlit
1.	Overview And OpenCV
2.	Face detection
3.	Mask Detection and save data
4.	Frontend
     

Training The Model
 
Python Code snippets:
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# DEFINE THE MODEL
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Flatten())
mymodel.add(Dense(100,activation='relu'))
mymodel.add(Dense(1,activation='sigmoid'))
mymodel.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
#define the data
train=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_img=train.flow_from_directory('train',target_size=(150,150),batch_size=16,class_mode='binary')
test_img=test.flow_from_directory('test',target_size=(150,150),batch_size=16,class_mode='binary')
# train and test the model
eye_model=mymodel.fit(train_img,epochs=10,validation_data=test_img)
mymodel.save('eye.h5',eye_model)
Backend
OpenCV:
Python code snippet:
Pip Install OpenCV
import cv2
vid=cv2.VideoCapture("http://192.168.277.91:8080/video")
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        cv2.nameWindow("Flag",cv2.WINDOW_NORMAL)
        cv2.imshow("Flag",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()
Eye Detection
 


Python Code Snippet:
import cv2
facemodel=cv2.CascadeClassifier("eye.xml")
vid=cv2.VideoCapture("face.mp4")
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)
        for(x,y,l,w) in faces:
            face_img=frame[y:y:w,x:x+l]
            cv2.rectangle(frame,(x,y),(x+l,y+w),(255,0,0),3)
        cv2.namedWindow("Drowsiness Detection",cv2.WINDOW_NORMAL)
        cv2.imshow("Drowsiness Detection",frame)
        key=cv2.waitKey(20)
        if(key==ord('x')):
            break       
    else:
        break
vid.release()
cv2.destroyAllWindows()






Closed Eye Detection & Saving Data
 
Python Code Snippet:
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
facemodel=cv2.CascadeClassifier("eye.xml")
eyemodel=load_model('eye.h5')
vid=cv2.VideoCapture("face.mp4")
i=1
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        face=facemodel.detectMultiScale(frame)
        for(x,y,l,w) in face:
            crop_face1=frame[y:y+w,x:x+l]
            cv2.imwrite('temp.jpg',crop_face1)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=eyemodel.predict(crop_face)[0][0]
            if pred==1:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(255,0,0),3)
                path="C:/Projects/eye/Scripts/data/"+str(i)+".jpg"
                cv2.imwrite(path,crop_face1)
                i=i+1
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,0),3)
        cv2.namedWindow("Drowsiness Detection",cv2.WINDOW_NORMAL)
        cv2.imshow("Drowsiness Detection",frame)
        k=cv2.waitKey(20)
        if(k==ord('x')):
            break
    else:
        break
cv2.destroyAllWindows()
Frontend
StreamLit:
 
Python Code Snipette:
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import tempfile
facemodel=cv2.CascadeClassifier("eye.xml")
eyemodel=load_model('eye.h5')
st.title("Drowsiness Detection System")
choice=st.sidebar.selectbox("My Menu",("HOME","IMAGE","VIDEO","CAMERA"))
if(choice=="HOME"):
    st.header("WELCOME")
elif(choice=="IMAGE"):    
    file=st.file_uploader("Upload Image")
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(img)
        for (x,y,h,w) in face:
            crop_face=img[y:y+w,x:x+h]
            cv2.imwrite('temp.jpg',crop_face)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=eyemodel.predict(crop_face)[0][0]
            if pred==1:
                cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),5)
            else:
                cv2.rectangle(img,(x,y),(x+h,y+w),(0,0,255),5)
        st.image(img,channels='BGR',width=400)
elif(choice=="VIDEO"):
    file=st.file_uploader("Upload Video")
    window=st.empty()
    btn=st.button("Stop Video")
    if btn:
        file=False
        st.experimental_rerun()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in face:
                    crop_face1=frame[y:y+w,x:x+l]
                    cv2.imwrite('temp.jpg',crop_face1)
                    crop_face=load_img('temp.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=eyemodel.predict(crop_face)[0][0]
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                        path="C:/Projects/eye/Scripts/data/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                window.image(frame,channels='BGR')
elif(choice=="CAMERA"):
    k=st.text_input("Enter 0 to open web cam or write url for opening ip camers")
    btn=st.button("Start Camera")
    window=st.empty()
    if btn:
        vid=cv2.VideoCapture(int(k))
        i=1
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in face:
                    crop_face1=frame[y:y+w,x:x+l]
                    cv2.imwrite('temp.jpg',crop_face1)
                    crop_face=load_img('temp.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=eyemodel.predict(crop_face)[0][0]
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                        path="C:/Projects/eye/Scripts/data/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                window.image(frame,channels='BGR')
    btn2=st.button("Stop Camera")
    if btn2:
        vid.close()
        st.experimental_rerun()




This will start the Streamlit server, and you can access the application in your web browser. The app will capture video from your webcam and process it to detect drowsiness, displaying alerts when drowsiness is detected.
