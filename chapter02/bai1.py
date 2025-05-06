import cv2
def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Left button clicked at (%d, %d)' % (x, y))
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        strxy = str(x) + ', ' + str(y)
        cv2.putText(img, strxy, (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[x,y,0]
        green = img[x,y,1]
        red = img[x,y,2]
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        strBGR = str(blue) + ', ' + str(grenn) + ', ' + str(red)
        cv2.putText(img,strBGR,(x,y),font,1,(0,255,255),2)
        cv2.imshow('image',img)
img = cv2.imread("C:/anh3.png")
cv2.imshow("anh 2",img)
cv2.setMouseCallback('image', click_event )
cv2.waitKey()
cv2.destroyAllWindow()