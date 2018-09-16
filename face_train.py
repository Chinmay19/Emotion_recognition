import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

base_path = "/media/chinmay/76D214DA706217A0/Abhyas/Practice/Face_Detection/"
data_path = "/media/chinmay/76D214DA706217A0/Abhyas/Practice/Face_Detection/training-data"
haar_face_cascade = cv.CascadeClassifier('/media/chinmay/76D214DA706217A0/Software/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml')


subjects = ["", "Palash", "Chinmay"]

def detect_face(haar_face_cascade, coloured_image, scale_factor):
	col_img = coloured_image.copy()

	gray_img = cv.cvtColor(col_img, cv.COLOR_BGR2GRAY)
	

	faces = haar_face_cascade.detectMultiScale(gray_img, scale_factor, minNeighbors = 5)

	if (len(faces) == 0):
		return None, None

	# save face parameters
	(x,y,w,h) = faces[0]

	# retun only those face parameteres.
	return gray_img[x:x+h, y:y+w], faces[0]

def prepare_training_data(data_folder_path):
	dirs = os.listdir(data_folder_path)

	faces = []
	labels= []

	for dir_name in dirs:
		label = int(dir_name.replace("s",""))
		print(label)

		subject_dir_path = data_folder_path +"/"+ dir_name
		subject_img_names = os.listdir(subject_dir_path)

		for image_name in subject_img_names:
			image_path = subject_dir_path +"/"+ image_name

			image = cv.imread(image_path)

			cv.imshow("training on image..", image)
			cv.waitKey(150)

			#detect_face
			face, rect = detect_face(haar_face_cascade, image, 1.1)
			if face is not None:
				faces.append(face)
				labels.append(label)

				cv.destroyAllWindows()
				cv.waitKey(1)
				cv.destroyAllWindows()

	return faces, labels


def draw_rectangle(img, rect):
	(x,y,w,h) = rect
	cv.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 2)

def draw_text(img, text, x, y):
	cv.putText(img, text, (x,y), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),2)

def predict(test_img):
	img = test_img.copy()
	face, rect = detect_face(haar_face_cascade, img, 1.1)

	label = face_recognizer.predict(face)
	label  = label[0]
	print("********************************")
	print (label)
	print("********************************")
	label_text = subjects[label]

	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)

	return img

print("Preparing data..")
faces, labels = prepare_training_data(data_path)
print("data prepared")

print("total faces: ",len(faces))
print("total labels: ",len(labels))

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


print("predicting images..")

test_img = cv.imread(base_path +"test-data/1.jpg")

predicted_img1 = predict(test_img)

print("done..")
print("predicting images..")

cv.imshow("1", predicted_img1)
cv.waitKey(0)
cv.destroyAllWindows()


test_img2 = cv.imread(base_path +"test-data/2.jpg")
predicted_img2 = predict(test_img2)
cv.imshow("2", predicted_img2)
cv.waitKey(0)
cv.destroyAllWindows()

print("done..")