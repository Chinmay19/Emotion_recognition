import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

path = "/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/"


# emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list
# emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions = ["anger", "sadness", "happy", "neutral", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
rf = RandomForestClassifier(n_estimators= 100)

 
data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(path + "/dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

# def get_landmarks(image):
#     landmarks = []
#     detections = detector(image, 1)
#     for k,d in enumerate(detections): #For all detected face instances individually
#         shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
#         xlist = []
#         ylist = []
#         for i in range(1,68): #Store X and Y coordinates in two lists
#             xlist.append(float(shape.part(i).x))
#             ylist.append(float(shape.part(i).y))
#         for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
#             landmarks.append(x)
#             landmarks.append(y)
#         data['landmarks'] = landmarks            
#     if len(detections) > 0:
#         pass
#     else: #If no faces are detected, return error message to other function to handle
#         data['landmarks'] = "error"
        # return landmarks
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
accur_lin = []

def test_live_stream():
    print("testing svm on live stream")
    cap = cv2.VideoCapture(0)
    # time.sleep(1)

    while (True):
        ret, frame = cap.read()
        test_frame = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_frame = clahe.apply(gray_frame)
        get_landmarks(clahe_frame)
        if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
        else:
            test_frame.append(data['landmarks_vectorised'])
            test_frame = np.array(test_frame)
            pred = rf.predict(test_frame)
            cv2.putText(img = frame, text = emotions[pred[0]], org=(100,100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3, color = (0,255,0))
            # print(emotions[pred[0]])
        cv2.imshow("frame", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()

for i in range(0,1):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("*******")
    print(prediction_labels, len(prediction_labels))
    print("******")
    print("training Random Forest Classifier %s" %i) #train SVM
    # clf.fit(npar_train, training_labels)
    rf.fit(npar_train, training_labels)
    # clf_grid.fit(npar_train, training_labels)
    # print("Best Parameters:\n", clf_grid.best_params_)
    # print("Best Estimators:\n", clf_grid.best_estimator_)    
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    prediction = rf.predict(prediction_data)

    print("*******")
    print(prediction, len(prediction)) 
    print("******")
    

    pred_lin = rf.score(npar_pred, prediction_labels)
    
    print ("random forest: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list

print("Mean value random forest: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs

# test_live_stream()