import cv2
import dlib
import numpy as np
from imutils import face_utils

path = "/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/"
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x,y,w,h)
    
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")
image = cv2.imread(path + "IMG_4559.jpg")
image = cv2.resize(image, (500,500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i,rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(image, "face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x,y) in shape:
        cv2.circle(image, (x,y), 1, (0,0,255), -1)

cv2.imshow("output", image)
cv2.waitKey(0)


