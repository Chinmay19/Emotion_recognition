import glob
from shutil import copyfile

path = "/media/chinmay/76D214DA706217A0/Abhyas/emotions_IOT/"
emotions = ["neutral", "anger", "contempt" ,"disgust", "fear", "happy", "sadness", "surprise"]

participants = glob.glob(path + "Source_emotion/*")

# print(participants)

for x in participants:
    part = "%s" %x[-4:]
    for sessions in glob.glob("%s/*" %x):
        for files in glob.glob("%s/*" %sessions):
            current_session = files[-33:-30]
            file = open(files, 'r')
            emotion = int(float(file.readline()))
            sourcefile_emotion = glob.glob(path +"Source_images/%s/%s/*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob(path +"Source_images/%s/%s/*" %(part, current_session))[0] #do same for neutral image
            dest_neut = path + "Sorted_set/neutral/%s" %sourcefile_neutral[-21:] #Generate path to put neutral image
            dest_emot = path + "Sorted_set/%s/%s" %(emotions[emotion], sourcefile_emotion[-21:]) #Do same for emotion containing image
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file
