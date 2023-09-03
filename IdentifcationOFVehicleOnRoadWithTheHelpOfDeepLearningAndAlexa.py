import zipfile
import warnings
import time
import cv2
import os
import random
import pyttsx3
import google
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout
from keras.layers import Conv2D,MaxPooling2D
from playsound import playsound



train_data_dir = "C:\\Users\\sony\\Downloads\\vechicleNonVechicle\\data"
test_data_dir =  "C:\\Users\\sony\\Downloads\\vechicleNonVechicle\\data"



datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.45
)


train_generator = datagen.flow_from_directory(
    train_data_dir,
    color_mode= "grayscale",
    target_size=(48,48),
    batch_size = 64,
    class_mode = "categorical",
    subset = 'training'
)

validation_generator = datagen.flow_from_directory(
    test_data_dir,
    color_mode= "grayscale",
    target_size=(48,48),
    batch_size = 64,
    class_mode = "categorical",
    subset = 'validation'
)


class_label = ['non-vehicles','vehicles']


print(f'Class_Label.{class_label}')

print(f'length of Class_Label.{len(class_label)}')


img,label = train_generator.__next__()



print(f'image In Matrix Form.{img}')


print('Model CNN Creation')

Dropout,
# Create a cnn model

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(48,48,1)))
model.add(Dropout(0.1))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Dense(100,activation="relu"))

model.add(Dense(2,activation="relu"))


print(f'Model_Summary.{model.summary()}')

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

print(voices[1].id)

engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

if __name__ == "__main__":

    speak("Hi Welcome In Image Processing Tutorial of Vehicle Detection On Road Side Detection  Using tensorflow . The Compilation of Model will Start Soon I Think You Love Our Hard Work and We Create Multiple Project of Machine Learning and Deep Learning In Future")


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

if __name__ == "__main__":
    speak('Training Is Started')

print('Training')
history=model.fit(train_generator,steps_per_epoch=25,epochs=10
                  ,validation_data=validation_generator)
if __name__ == "__main__":
    speak('Training Is Complete You Can See The Loasses And Accuracy Plot of Our Model')


print('Accuracy and Loasses Plot')


history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()

plt.show()


#TEST OUR DATASET

import numpy as np
import pandas as pd
import os
import cv2
from keras.models import load_model
#import winsound


labels_dict = {0:'non-vehicles',1:'vehicles'}

frame = cv2.imread("C:\\Users\\sony\\Downloads\\vechicleNonVechicle\\data\\vehicles\\3581.png")

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


#sub_face_img = gray[y:y+h, x:x+w]
resized = cv2.resize(gray,(48,48))
normalize = resized/255.0
reshaped = np.reshape(normalize,(1,48,48,1))
result = model.predict(reshaped)
label = np.argmax(result,axis=1)[0]
print(voices[1].id)

engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

if __name__ == "__main__":
   
   if (label==1):
      playsound("C:\\Users\\sony\\Downloads\\emergency-alarm-with-reverb-29431.mp3")
      
   

   while(True):
     
      
     if (label==0):
       speak("No Vehicles Seen In Car Frame You Continue With Same Speed")
     elif (label==1):

       speak("Please Maintain Your Speed Vechicle Are Seen In Frame")
plt.imshow(img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()