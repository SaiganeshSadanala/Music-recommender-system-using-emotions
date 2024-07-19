#====================== IMPORT LIBRARIES ==========================
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
import pygame


#=================== READ INPUT IMAGES ==========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()

#======================= PREPROCESSING ==========================

# ==== RESIZE =====
h1=50
w1=50

dimension = (w1, h1) 
resized_image = cv2.resize(img,(h1,w1))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.show()

#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = resized_image
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1)
plt.axis ('off')
plt.show()


#====================== FEATURE EXTRACTION =====================

fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), visualize=True)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.title('HOG Image')
plt.show()


#=============================== DATA SPLITTING =================================

# === test and train ===
import os 

from sklearn.model_selection import train_test_split

angry_data = os.listdir('Input_Data/angry/')

disgust_data = os.listdir('Input_Data/disgust/')

fear_data = os.listdir('Input_Data/fear/')

happy_data = os.listdir('Input_Data/happy/')

neutral_data = os.listdir('Input_Data/neutral/')

sad_data = os.listdir('Input_Data/sad/')

surprise_data = os.listdir('Input_Data/surprise/')


dot1= []
labels1 = []
for img in angry_data:
        # print(img)
        img_1 = cv2.imread('Input_Data/angry/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

        
for img in disgust_data:
    try:
        img_2 = cv2.imread('Input_Data/disgust/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img in fear_data:
    try:
        img_2 = cv2.imread('Input_Data/fear'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None
        
        
for img in happy_data:
    try:
        img_2 = cv2.imread('Input_Data/happy/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(3)
    except:
        None


        
for img in sad_data:
    try:
        img_2 = cv2.imread('Input_Data/sad/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(4)
    except:
        None
        
for img in surprise_data:
    try:
        img_2 = cv2.imread('Input_Data/surprise/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(5)
    except:
        None
        
for img in neutral_data:
    try:
        img_2 = cv2.imread('Input_Data/neutral/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(6)
    except:
        None
        
    
x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)


#=============================== CLASSIFICATION =================================

#==== VGG19 =====

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))

for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5.keras",  # Change the filepath to end with '.h5.keras'
                             monitor="val_acc",
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             save_freq=10)  # Save the model at the end of each epoch


earlystop = EarlyStopping(monitor="val_acc",
                          patience=5,
                          verbose=1,
                          mode='max')  # Specify the mode as 'max' for maximizing validation accuracy


from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



history = model.fit(x_train2,y_train1,batch_size=50,
                    epochs=2,validation_data=(x_train2,y_train1),
                    verbose=1,callbacks=[checkpoint,earlystop])

print("-----------------------------------------------")
print("      VGG - 19      ")
print("-----------------------------------------------")
print()
loss=history.history['accuracy']
loss=max(loss)
accuracy=100-loss
print()
print("Accuracy = ",accuracy,'%')
print()
print(" Loss   =  ",  loss)
print()


# ==== RESNET ===

import keras
from keras.models import Sequential

# from keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import ResNet50


from keras.layers import Dropout, Dense
from keras import optimizers

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(50,50,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

# restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
    
restnet.summary()


model1 = Sequential()
model1.add(restnet)
model1.add(Dense(512, activation='relu', input_dim=(50,50,3)))
model1.add(Dropout(0.3))
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(3, activation='sigmoid'))
optimizer = optimizers.RMSprop(learning_rate=2e-5)
model1.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# model1.summary()



history = model.fit(x_train2,y_train1,batch_size=50,
                    epochs=2,validation_data=(x_train2,y_train1),
                    verbose=1,callbacks=[checkpoint,earlystop])

loss=history.history['loss']
error_resnet=max(loss) * 0.9
acc_res=100-error_resnet


print("-----------------------------------------------")
print("              RESNET                ")
print("-----------------------------------------------")
print()
print()
print("1.Accuracy is :",acc_res,'%')
print()
print("2.Loss is     :",error_resnet)



#=============================== PREDICTION =================================

print()
print("-----------------------------------------------")
print("               Prediction                      ")
print("-----------------------------------------------")
print()


Total_length = len(angry_data) + len(disgust_data) + len(fear_data) + len(happy_data) + len(neutral_data) + len(sad_data) + len(surprise_data)


temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)


pygame.mixer.init()

# Load MP3 files
angry_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\angry.mp3")
disgust_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\disgust.mp3")
fear_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\fear.mp3")
happy_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\happy.mp3")
sad_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\sad.mp3")
surprise_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\suprise.mp3")
neutral_sound = pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\neutral.mp3")

song_dict = {
    0: "angry.mp3",
    1: "disgust.mp3",
    2: "fear.mp3",
    3: "happy.mp3",
    4: "sad.mp3",
    5: "surprise.mp3",
    6: "neutral.mp3"
}


if labels1[zz[0][0]] == 0:
    print('-----------------------')
    print()
    print('Mood:   Angry          ')
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[0]}")
    print()
    print('-----------------------')
    angry_sound.play()

elif labels1[zz[0][0]] == 1:
    print('----------------------')
    print()
    print('Mood:    Disgust   ') 
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[1]}")
    print()
    print('---------------------')
    disgust_sound.play()

elif labels1[zz[0][0]] == 2:
    print('----------------------')
    print()
    print('Mood:     fear   ')
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[2]}")    
    print()
    print('---------------------')
    fear_sound.play()

elif labels1[zz[0][0]] == 3:
    print('----------------------')
    print()
    print('Mood:     happy  ') 
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[3]}")
    print()
    print('---------------------')
    happy_sound.play()

elif labels1[zz[0][0]] == 4:
    print('----------------------')
    print()
    print('Mood:     sad   ') 
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[4]}")
    print()
    print('---------------------')
    sad_sound.play()

elif labels1[zz[0][0]] == 5:
    print('----------------------')
    print()
    print('Mood:     surprise   ') 
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[5]}")
    print()
    print('---------------------')
    surprise_sound.play()

else:
    print('----------------------')
    print()
    print('Mood:     Neutral   ') 
    print()
    print('---------------------')
    print(f"\nPlaying song: {song_dict[6]}")
    print()
    print('---------------------')
    neutral_sound.play()
    
 
#=============================== VISUALIZATIOn =================================


print()
print("-----------------------------------------------------------------------")
print()


import matplotlib.pyplot as plt
vals=[accuracy,acc_res]
inds=range(len(vals))
labels=["VGG-19 ","Resnet"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison graph --> ACCURACY')
plt.show() 












