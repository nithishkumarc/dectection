#!C:\Python311\python.exe
import cgi;
import mysql.connector;
import pywhatkit as kit;
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from keyboard import press

print('Content-type:text/html\n\n')
f = cgi.FieldStorage();
userid= f.getvalue('user_name');
query = "select * from user where userid='{usrid}'".format(usrid=userid);
db = mysql.connector.connect(host="localhost",db="nec",user="root",password="");
cursor = db.cursor();
cursor.execute(query);
user_data = cursor.fetchall();
user_name = mobile1 = mobile2 = address = "";
for user_data1 in user_data:
    user_name = user_data1[1];
    mobile1 = user_data1[2];
    mobile2 = user_data1[3];
    address = user_data1[4];    
db.commit();

mobile_lst = [mobile1,mobile2];





mixer.init()
sound = mixer.Sound('alarm.wav')
sound1 = mixer.Sound('Theme1.mp3')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        #rpred = model.predict(r_eye)
        rpred = np.argmax(model.predict(r_eye),axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        #lpred = model.predict_classes(l_eye)
        lpred = np.argmax(model.predict(l_eye),axis=-1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            for mobile in mobile_lst:
                message = "Please rescue your friend and drop into this address {addr}".format(addr = address);
                x = datetime.datetime.now();
                kit.sendwhatmsg("+91"+mobile,message,int(x.strftime("%H")),int(x.strftime("%M"))+3,59);
                press('enter');
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    #else:
        #time.sleep(30)
        #sound1.play();
    cv2.imshow('frame',frame)   
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()











    


