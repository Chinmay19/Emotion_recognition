import cv2
import glob
import random
import numpy as np
import time
emotions = ["neutral", "anger", "disgust", "happy", "surprise"] #Emotion list
fishface = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier
faceDet = cv2.CascadeClassifier('/home/chinmay/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

data = {}
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    
    print ("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
#Now run it

def test_live_stream():
    print("testing fisher face on live stream")
    cap = cv2.VideoCapture(0)
    # time.sleep(1)

    while (True):
        ret, frame = cap.read()
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceDet.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face) > 0:
            (x,y,w,h) = face[0]
            img = gray_frame[y:y+h, x:x+w]
            img = cv2.resize(img, (350,350))    
            pred, conf = fishface.predict(img)
            cv2.putText(img = frame, text = emotions[pred], org=(300,384), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3, color = (0,255,0))
            print(emotions[pred])
        else:
            pass
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()




metascore = []
for i in range(0,2):
    correct = run_recognizer()
    print ("got", correct, "percent correct!")
    metascore.append(correct)
print ("\n\nend score:", np.mean(metascore), "percent correct!")
test_live_stream()
