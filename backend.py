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
