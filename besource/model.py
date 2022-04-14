######################
##  IMPORT ALL LIB  ##
######################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet201

######################
##  PRE PROCESSING  ##
######################
df = pd.read_csv('./english.csv')
#Get all image
directory = './Img'
files=os.listdir(directory)
datafile=[]
data=[]
#Image to array with reshape image to 96x96
for file in files:
    image=load_img(os.path.join(directory,file),grayscale=False,color_mode='rgb',target_size=(96,96))
    image=img_to_array(image)
    image=image/255.0
    data+=[image]
    datafile+=[file]
data = np.array(data)
#Categorical
label, uniques = pd.factorize(df['label'])
#Set Feature Img/ to only /
labels_of_file = []
for item in df['image']:
    labels_of_file+=[item[4:]]
df['file']=labels_of_file
df['labell']=label
#Label to categorical
data_new_label=[]
for item in datafile:
    data_new_label+=[df['labell'][df['file']==item].values[0]]
labels=to_categorical(data_new_label)
labels=np.array(labels)
#Split Data
train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=42)
######################
##       MODEL      ##
######################
datagen = ImageDataGenerator(rotation_range=20,
                            width_shift_range=0.2, 
                            height_shift_range=0.2, 
                            shear_range=0.1, 
                            zoom_range=0.2,
                            validation_split=0.35,
                            fill_mode='nearest')
datagen.fit(train_x)

base_model = DenseNet201(input_shape=(96,96,3), weights='imagenet', include_top=False)
model = tf.keras.Sequential([base_model,
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(62, activation='softmax')
                             ])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)
history = model.fit(
    datagen.flow(train_x, train_y, batch_size=32,subset='training'),
    validation_data=datagen.flow(train_x, train_y,batch_size=8, subset='validation'),
    epochs=35,
    callbacks=[callback]
)

######################
##      TESTING     ##
######################
pred = model.predict(test_x)
classDict = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A",
            11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
            21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
            31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "c", 39: "d", 40: "e",
            41: "f", 42: "g", 43: "h", 44: "i", 45: "j", 46: "k", 47: "l", 48: "m", 49: "n", 50: "o",
            51: "p", 52: "q", 53: "r", 54: "s", 55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y",
            61: "z"}
maxIndex = list(pd.DataFrame(pred).idxmax(axis=1))
for i in range(15):
    plt.title(classDict.get(maxIndex[i]))
    # Show the actual image
    plt.imshow(test_x[i])
    plt.show()

#######
#SAVE #
#######

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")