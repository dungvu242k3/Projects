import cv2
import numpy as np
img = np.ones((512,512,3),np.uint8)*255
colors = [(0, 0, 0),   
    (0, 0, 255),     
    (0, 255, 0),     
    (0, 0, 100),     
    (173, 150, 200),  
    (255, 182, 193), 
    (255, 255, 0),   
    (255, 0, 255)]
for i,color in enumerate(colors):
    img[i*64:(i+1)*64,412:512] = color
def get_pixel_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y,x]
        print("bgr : ",bgr)
        rgb = tuple(reversed(bgr))
        print('rgb',rgb)
    if event == cv2.EVENT_MOUSEMOVE:
        ol = img[y,x]
        print('lol',lol)
cv2.namedWindow('image')
cv2.setMouseCallback("image",get_pixel_color)
while True :
    cv2.imshow('image',img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
    