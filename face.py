import cv2 as cv


source_path = "-add base directory path"
haar_face_cascade = cv.CascadeClassifier('--cv2 package location--/data/haarcascade_frontalface_alt.xml')


def detect_Faces(haar_face_cascade, coloured_img, scaleFactor):
	col_img = coloured_img.copy()
	gray_img = cv.cvtColor(col_img, cv.COLOR_BGR2GRAY)

	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighbors =5)
	return (faces)


def draw_rectanges(coloured_img, faces):

	for (x,y,w,h) in faces:
		cv.rectangle(coloured_img, (x,y), (x+w,y+h), (0,0,255), 2)
	

	cv.imshow('Faces_Found', coloured_img) 
	cv.waitKey(0) 
	cv.destroyAllWindows()


input_img = cv.imread(source_path + "data/multi_test.jpg" )
faces = detect_Faces(haar_face_cascade, input_img, 1.1)
draw_rectanges(input_img, faces)







