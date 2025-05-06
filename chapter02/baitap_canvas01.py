import cv2
import numpy as np

img = np.ones((512, 512, 3), dtype=np.uint8) * 255

colors = [
    (0, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 0, 100),
    (173, 150, 200),
    (255, 182, 193),
    (255, 255, 0),
    (255, 0, 255)
]

for i, color in enumerate(colors):
    img[i * 64:(i + 1) * 64, 412:512] = color

drawing = False
mode = True
color= (0, 0, 0)

def click(event, x, y, flags, param):
    global drawing, mode, color

    if 412 < x < 512 and 0 <= y < 512:
        color = img[y, x].tolist()

    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if mode:
            cv2.rectangle(img, (x, y), (x, y), color, 10)
        else:
            cv2.circle(img, (x, y), 10, color, 10)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(img, (x, y), (x, y), color, 10)
            else:
                cv2.circle(img, (x, y), 10, color, 10)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, (x, y), (x, y), color, 10)
        else:
            cv2.circle(img, (x, y), 10, color, 10)

cv2.namedWindow("image")
cv2.setMouseCallback('image', click)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
