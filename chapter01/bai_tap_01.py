import numpy as np
import cv2
img = np.ones((512,512, 3), dtype=np.uint8) * np.array([0, 200, 200], dtype=np.uint8)
color = (0,200,255)
thinkness = -1
cv2.rectangle(img,(0,0),(130,130),color,thinkness)
cv2.rectangle(img,(382,382),(512,512),color,thinkness)
cv2.rectangle(img,(0,512),(130,382),color,thinkness)
cv2.rectangle(img,(512,0),(382,130),color,thinkness)
cv2.circle(img,(256,256),178,(255,132,255),2)
cv2.arrowedLine(img,(256,256),(130,130),(0,0,255),5)
cv2.arrowedLine(img,(256,256),(382,382),(0,0,255),5)
cv2.arrowedLine(img,(256,256),(382,130),(0,0,255),5)
cv2.arrowedLine(img,(256,256),(130,382),(0,0,255),5)
pts = np.array([[256,0],[512,256],[256,512],[0,256]],np.int32)
cv2.polylines(img,[pts],True,(255,255,0),3)
cv2.ellipse(img,(256,39),(126,38),0,0,360,(255,0,0),2)
cv2.ellipse(img,(39,256),(38,126),0,0,360,(255,0,0),2)
cv2.ellipse(img,(256,473),(126,38),0,0,360,(255,0,0),2)
cv2.ellipse(img,(473,256),(38,126),0,0,360,(255,0,0),2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,"Area 1",(40,70),font,0.6,(0,255,255),1,cv2.LINE_AA)
cv2.putText(img,"Area 2",(422,70),font,0.6,(0,255,255),1,cv2.LINE_AA)
cv2.putText(img,"Area 4",(40,452),font,0.6,(0,255,255),1,cv2.LINE_AA)
cv2.putText(img,"Area 3",(422,452),font,0.6,(0,255,255),1,cv2.LINE_AA)

cv2.imshow("dung vu",img)
cv2.waitKey()
cv2.destroyAllWindow()

