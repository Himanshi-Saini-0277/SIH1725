import cv2
import os

video = cv2.VideoCapture(0)

imageDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

nameID = str (input("Enter Day number: ")).lower()
path = r'C:\Users\himan\Documents\SIH1725\Images\{}'.format(nameID)

Exist = os.path.exists(path)

if (Exist):
    print("Name Already Taken")
    nameID = str (input("Enter Day Number Again: "))
else:
    os.mkdir(path)
    
while True:
    ret, frame = video.read()
    faces = imageDetect.detectMultiScale(frame, 1.3,5)
    for x,y,w,h in faces:
        count = count+1
        name = './images/' + nameID + '/' + str(count) + '.jpg'
        print("Creating images: " +name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        
    cv2.imshow("Window Frame", frame)
    if count>10:
        break
    
video.release()
cv2.destroyAllWindows()