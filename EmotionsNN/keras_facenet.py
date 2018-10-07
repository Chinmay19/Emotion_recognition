import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


import itertools
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard


config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 56} )
sess = tf.Session(config = config)
keras.backend.set_session(sess)

MODEL_NAME = "keras-facenet-cnn-64x2-128x1-{}".format(int(time.time()))

tensorBoard = TensorBoard(log_dir='/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/EmotionsNN/logs/{}'.format(MODEL_NAME))

num_classes = 7 # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 64
epochs = 20

with open("/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/EmotionsNN/data/fer2013/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print ("number of examples: ", num_of_instances)
print ("instance length: ", len(lines[2].split(",")[1].split(" ")))


x_train, y_train, x_test, y_test = [],[],[],[]

for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
        # val = img.split(" ")
        pixels = np.array(img.split(" "), 'float32')
        # pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)

        if 'Training' in usage:
            x_train.append(pixels)
            y_train.append(emotion)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255
x_test /= 255


x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()

#layer 1
model.add(Conv2D(64,(5,5), activation = 'relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2)))

#layer 2
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))

#layer 3
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides = (2,2)))

model.add(Flatten())

# fully connected network
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation = 'softmax'))

gen = ImageDataGenerator()
# train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
gen.fit(x_train)
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

his = model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, callbacks = [tensorBoard])
# model.fit_generator(train_generator, steps_per_epoch = batch_size, epochs = epochs)
model.save_weights('/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/EmotionsNN/keras_facenet_weights2.h5')
model.save('/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/EmotionsNN/keras_facenet2')
print('model and weights saced successfully\n')
print("---------------------------------------------")
print (his.history)
print("---------------------------------------------")
# def emotion_analysis(emotions):
#     objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#     y_pos  = np.arange(len(objects))

#     plt.bar(y_pos, emotions, align = 'center', alpha=0.5)
#     plt.xticks(y_pos, objects)
#     plt.ylabel('percentage')
#     plt.title('emotion')

#     plt.show()

predictions = model.predict_classes(x_test, batch_size=64)
scores = model.evaluate(x_test, y_test, verbose=1)
print("==========================================================")
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])
print("==========================================================")

# cm = confusion_matrix(y_test, predictions)
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

# cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')

# index = 0
# for i in predictions:
#     if index < 30 and index >= 20:
#         testing_img = np.array(x_test[index], 'float32')
#         testing_img = testing_img.reshape([48,48])

#         plt.gray()
#         plt.imshow(testing_img)
#         plt.show()
#         print(i)

#         emotion_analysis(i)
#         print("******=======================******")
#     index = index+1