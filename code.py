### Implemented by Mohsin Ahmed (mohsinahmednow@gmail.com) and Prerit S Mittal (prerit.mittal29@gmail.com)

## **Data Preparation**

###Handling all imports
"""

import tensorflow as tf 
import keras
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
import datetime, os
import math
from keras import backend

from pathlib import Path
from random import shuffle
from pylab import rcParams 
import matplotlib.pyplot as plt
import time

import cv2
import numpy as np
import os
import pickle
import pandas as pd
import zipfile
import math
from collections import defaultdict

!pip install openpyxl==3.0.0

"""### **Mount GoogleDrive to Google Colab**"""

from google.colab import drive
drive.mount('/content/drive')

!ls ./drive/MyDrive/PenmanshipProject/HKCollege

from google.colab import drive
drive.flush_and_unmount()

"""### **Unziping data online**"""

local_zip = '/content/drive/MyDrive/PenmanshipProject/HKCollege/HKCData.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/PenmanshipProject/HKCollege')
#the data has been unzipped in the colab driver but it won't be successfully unzipped in Google driver

"""### **Mount Onedrive to Google Colab**"""

! curl https://rclone.org/install.sh | sudo bash

!rclone config

!mkdir Onedrive
!nohup rclone --vfs-cache-mode writes mount Onedrive: ./Onedrive &   
#make sure the name "Onedrive" should be identical with the name you created in the config: Case sensitive

!ls ./Onedrive 
#check if the clound driver has been successfully loaded 
# If you want to unmount driver: Click Runtime -- Factory reset runtime  if got 'cannot access './Onedrive': Transport endpoint is not connected'

"""### **Score Data Visualization**"""

from matplotlib import rcParams
rcParams['figure.figsize'] = 25,10

excelSheetPath='/content/Onedrive/PenmanshipProject/HKCPenData/Data/AllDataSummary.xlsx'  # excelSheetPath is equal to the excel file

Fig = pd.read_excel(excelSheetPath, sheet_name = 'AvePenmanshipData')

Fig.boxplot(by = 'Subject',column = 'AverageScores1')

"""###**Image Data Loading & Pre-processing ( start from "Pickle files Loading" if you have already converted the arrays into a pickle file)**"""

imgFolder='Image Folder path'

fol_list=sorted(os.listdir(imgFolder)) # List the folder names in the imgFolder --> hk01, hk02 ...

drip=defaultdict(list)   # output the 'drip' value with a list format
for i in fol_list:   # loop the item names in fol_list
    img_path=imgFolder+'/'+i+'/'   

    for j in sorted(os.listdir(img_path)): 
        drip[i].append(j[:-4])

"""### **Preprocessing Function**"""

def resize_image(fol,im):  
    imgFolder='/content/Onedrive/PenmanshipProject/MLCPenData/Data/MLCDataTrain/'

    img = cv2.imread(imgFolder+fol+'/'+im, cv2.IMREAD_GRAYSCALE)  
    thresh = 127 
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]  # convert to black and white image
    
    scale_percent = 20         # define the image width and height and save to 'dim' varibale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)  # Use cv2.resize function to convert the image with the design features (as mentioned above)
    resized = np.asarray(resized)                   # use np.asarray function to convert the image to numpy array      
    w,h = resized.shape[1],resized.shape[0]
    x = h//2     # The “//” operator is used to return the closest integer value which is less than or equal to a specified expression or value. So 5//2 returns 2. You know that 5/2 is 2.5, the closest integer which is less than or equal is 2[5//2].( it is inverse to the normal maths, in normal maths the value is 3).
    resized = np.asarray(resized)[0:h, w//2-x:w//2+x]  
    return resized

"""### **Convert the images into arrays** """

excelSheetPath='/content/Onedrive/PenmanshipProject/HKCPenData/Data/AllDataSummary.xlsx'

excel_sheet=pd.read_excel(excelSheetPath,sheet_name=['AvePenmanshipData'])

total_pics=len(excel_sheet['AvePenmanshipData']['Pic']) # len function is to count the number of lists from 'pic'
data=defaultdict(list)       # output the data value with a list format         
for i in range(total_pics):   
    pic_name=excel_sheet['AvePenmanshipData']['Pic'][i]  # loop and save all the pic names from 'pic' in the pic_name variable; [i] to loop
    ave_score=excel_sheet['AvePenmanshipData']['AverageScores1'][i]  # loop and save all the scores/values from 'AverageScores1' in the ave_score variable; [i] to loop
    fol=pic_name[:4]    
                    
    if math.isnan(ave_score) or (pic_name not in drip[fol]):  # 'drip' was defined above, to utput the 'drip' value with a list format; if ave_score is a NA, or pic_name not in drip[fol] (hk01, hk02...) 
        continue                        
    img=resize_image(fol,pic_name+'.png')    # img contains all the images in a array format
    data[fol].append([img,ave_score,pic_name])  # 'data' has been defined to output the data in a list format; '[fol]' is to give the list name based on 'hkxx', will give all the array in a list, but in different hkxx 'folder'; 
print("Converted")

"""### **Convert the arrays into a Pickle file**"""

print('Saving Data')
pickle_file = '/content/Onedrive/PenmanshipProject/MLCPenData/PickleFiles/MLCDataTrain.pickle'
pickle.dump(data,open(pickle_file,'wb'))  # save the data into a pickle file format

"""### **Pickle file Loading**"""

data_path = '/content/Onedrive/PenmanshipProject/HKCPenData/PickleFiles/HKCData.pickle'

raw_data = pickle.load(open(data_path,'rb'))  # assign the pickle data in 'raw_data'
training_data = []   
for i in raw_data:
  for j in raw_data[i]:  
    training_data.append([j[0],j[1],j[2]]) # got arrays, scores and item, so we can make sure item names and their scores are not mixed 
print("picklefile loaded")

"""### **Dataset Loading**"""

#We already had the pickle file of processed images so we will augment from those only to increase efficiency
data_path='/content/Onedrive/PenmanshipProject/HKCPenData/PickleFiles/HKCData.pickle' # pickle file path of processed images
file = open(data_path,'rb')
raw_data = pickle.load(file)

# Checking the number of images for each of the scores
nums=[0]*10
for auth in raw_data:
    for im in range(len(raw_data[auth])):
        nums[round(raw_data[auth][im][1])]+=1
print(nums)

new_data=defaultdict(list)
for auth in raw_data:
    for im in range(len(raw_data[auth])):
        score=round(raw_data[auth][im][1])
        new_data[score].append(raw_data[auth][im])

# This is will create a pickle file of the dictionary for later uses, we can directly use the dictionary created before too so this might be optional
pre="/content/drive/MyDrive/Penmanship_project/Latest_data/pickeled_data_by_scores_240/"
post='_image_score_imName240bin.pickle'

for score in new_data:
    temp=new_data[score]
    print(len(temp))
    pickle_name=pre+str(score)+post
    pickle.dump(temp,open(pickle_name,'wb'))
print("Finished")

"""### **Data Augmentation**

The following function (make_more_images) is the augmentation function created to perform the following tasks :-
1. Take snapshots of the images from different viewpoints (ie. rotation, shifting, zooming)
2. Saving the images at the below mentioned folders using npp and pdp (file paths) according to the writers and scores
"""

npp="" #Path of folder where you want to save the new images
pdp="" #Path of pickled dictionary (the one created in above steps), in case of not creating the pickle, we can directly use the above object

def make_more_images(num_of_images,fol_name,arr):
    im_shape=240 # change this in case the dimension of the images are different
    datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1,
            fill_mode='nearest')
    for y in arr:
        img=y[0]
        score=y[1]
        name=y[2]+'__'+str(score)+'_' #the name of the augmented image
        img=img.reshape((im_shape,im_shape,1))
        img=img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size=32,save_to_dir=npp+fol_name+'/', save_prefix=name, save_format='png'):
            i += 1
            if i > num_of_images:
                break  # otherwise the generator would loop indefinitely

sc='1' # in case you generated the pickle files you have to change this to the score whose images you want to augment
pick=sc+'_image_score_imName240bin.pickle'
f=pdp+pick
file = open(f,'rb')
arr = pickle.load(file)
n=math.ceil(5000/len(arr))
print(pick,pick[0],n)
make_more_images(n,pick[0],arr)

"""### **Making new pickle file from the augmented images**"""

aug_data_fol="" # path of the augmented images folder
new_pkl_loc="" # path of the folder where the final file should be save

data=defaultdict(list)
culp=['1','2','7','8','9']

for fol in culp:
    for im_name in os.listdir(aug_data_fol+fol):
        s=im_name.split("__")
        name=s[0]
        auth=name[0:4]
        score=float(s[1])
        img=cv2.imread(aug_data_fol+fol+'/'+im_name,cv2.IMREAD_GRAYSCALE)
        temp=[img,score,name,auth]
        data[round(score)].append(temp)
    print(fol,'finished')

num_of_Images_to_clip=5000 
for auth in data:
    data[auth]=data[auth][0:num_of_Images_to_clip]

# We now convert the images into the previous format of original data
new_data=defaultdict(list)
for auth in data:
    for im in data[auth]:
        new_data[im[3]].append(im[0:3])

# We now finally create the pickle file containg all the new images we will then combine this with our original data
pickle_file_name="new_augmented_image_score_imName240bin.pickle"
pickle_name=new_pkl_loc+pickle_file_name
pickle.dump(new_data,open(pickle_name,'wb'))

"""### **Data Shuffling**"""

#shuffle the data. Right now our scores can be just all 4 or 5, This will usually wind up causing trouble too, as, initially, the model will learn to just predict scores from hk1. Then it will shift to oh, just predict from hk02 to hkxx! Going back and forth like this is no good either.
import random 

random.shuffle(training_data) 

for sample in training_data[:5]: #double-check the shiffled data
  print(sample)

"""## **Training the model**

"""

# This code in the 'for' loop is to segregate the data into training(train) and validation(test) set. 
# from raw_data (the pickle file created of the orginal dataset after data preparation for original dataset)
# Note that raw_data is already shuffled

for i in raw_data:               
    l=len(raw_data[i])
    x=round(0.9*l)
    for j in range(x):
        train.append([raw_data[i][j][0], raw_data[i][j][1]])
    for j in range(x,l):
        test.append([raw_data[i][j][0],  raw_data[i][j][1]])

del raw_data
del aug_data

# shuffling the training (train) and validation(test) set
shuffle(train)
shuffle(test)

# Finally creating X_train (Training Image Array), Y_test (train true values),
# X_test (Validation Image Array), Y_test (validation true values)
for i in train:
    X_train.append(i[0])
    Y_train.append((i[1]))
for i in test:
    X_test.append(i[0])
    Y_test.append((i[1]))
del train
del test

# Creating Numpy Array for all (compatible format)
X_train,Y_train,X_test,Y_test=np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)

# Reshaping the Image arrays into 3d Image array (Last dimension 1 - B/W)

image_size=X_train.shape[-1]
train_size, test_size=X_train.shape[0],X_test.shape[0]
X_train=(X_train.reshape(train_size, image_size, image_size, 1))
X_test=(X_test.reshape(test_size, image_size, image_size, 1))

# Normalizing the Image Arrays
X_train=X_train//255
X_test=X_test//255

# Final Image Arrays Dimensions (datapoints,image length, image breadth, color channels (1- B/W , 3 - RGB))
print(X_train.shape)
print(X_test.shape)

"""### Model Performance Metrics
"""

def MET(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  diff = tf.abs(
    (y_true - y_pred) )
  Pmax = 9.33 # Maximum penmanship score
  Pmin = 1  # Minimum penmanship score
  temp = 100/(Pmax-Pmin) 
  return temp*backend.mean(diff, axis=-1)

"""## Building the Model"""

def model_blueprint(filter1=16, filter2=32, filter3=64, ker_size1=3,ker_size2=5,ker_size3=5, node1=510, node2=480, node3=350, node4=10):
    dropout=0.5
    LOSS=tf.keras.losses.MeanSquaredError() 
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=1e-3, decay_steps=5, decay_rate=0.3, staircase=True, name=None)
    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    model = Sequential([ 
        Conv2D(filters=filter1, kernel_size=(ker_size1,ker_size1),activation='relu', input_shape = (image_size, image_size, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3)),
        
        Conv2D(filters=filter2, kernel_size=(ker_size2,ker_size2), padding='same',activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3)),

        Conv2D(filters=filter3, kernel_size=(ker_size3,ker_size3),padding="same", activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3)),

        Dropout(dropout),
        Flatten(),

        Dense(node1,activation='relu', kernel_initializer= tf.keras.initializers.HeNormal(),kernel_regularizer=keras.regularizers.l1()),
        Dropout(dropout),
        Dense(node2,activation='relu', kernel_initializer= tf.keras.initializers.HeNormal(),kernel_regularizer=keras.regularizers.l2(0.001)),
        Dense(node3,activation='relu', kernel_initializer= tf.keras.initializers.HeNormal(),kernel_regularizer=keras.regularizers.l1_l2(0.01)),
        Dropout(dropout),
        Dense(node4,activation='relu', kernel_initializer= tf.keras.initializers.HeNormal(),kernel_regularizer=keras.regularizers.l1_l2(0.01)),
        Dense(1,activation='relu')
    ])
    model.compile(loss=LOSS, optimizer=opt,metrics=[MET])
    return model

# Building the Model using function call
model=model_blueprint(filter1=16, filter2=32, filter3=64, ker_size1=3,ker_size2=5,ker_size3=5, 
                      node1=510, node2=480, node3=350, node4=10)
# The variables values are set to our proposed model
# model.summary()

"""### Model Training"""

# Commented out IPython magic to ensure Python compatibility.
def model_fit(model,X_train,Y_train,X_val,Y_val,batch_size=32,epochs=70,verbose=1,callbacks=None):
    history=model.fit(
        x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
        callbacks=callbacks, validation_data=(X_test,Y_test), shuffle=True,use_multiprocessing=False
        )
    return history

# %load_ext tensorboard

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

NAME= "Penmanship-cnn-adam-{}".format(int(time.time())) # Customizable
print(NAME)
history=model_fit(model,X_train=X_train,Y_train=Y_train,X_val=X_test,Y_val=Y_test,callbacks=[tensorboard_callback])

# %tensorboard --logdir logs

"""# Testing with various methods

### Relaxed nMAPE

Custom metric to measure the accuracy of the model, realxed MAPE is calculated by taking the MAPE of predicted score, predicted score+1 and predicted score-1
"""

# Custom metric definition
def MET(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  diff1 = tf.abs((y_true - y_pred))
  diff2 = tf.abs(y_true - (y_pred-1))
  diff3 = tf.abs(y_true - (y_pred+1))
  temp=tf.minimum(backend.mean(diff1, axis=-1),backend.mean(diff2, axis=-1))
  temp=tf.minimum(temp,backend.mean(diff3, axis=-1))
  return (10.0*temp)

# Preparing input for testing
x=defaultdict(list) #Images of characters
y=defaultdict(list) #scores

#Adding data from original archive
for i in raw_data:
    l=len(raw_data[i])
    for j in range(l):
        score=round(raw_data[i][j][1])
        x[score].append(raw_data[i][j][0])
        y[score].append(raw_data[i][j][1])

#Preprocessing the data
for i in range(1,10):
  x[i]=np.array(x[i])
  image_size=x[i].shape[-1]
  train_size=x[i].shape[0]
  x[i]=(x[i].reshape(train_size, image_size, image_size, 1))
  x[i]=x[i]//255
  y[i]=np.array(y[i])

#Evaluating for each class of score
for i in range(1,10):
    y_pred=model.predict(x[i])
    history = MET(y[i],y_pred)
    a=sum(history)/len(x[i])
    print("Percentage error for score ",i," :",end=" ")
    tf.print(a)
print("---------------------------------------------------")

"""### Checking for individual penmanship score class"""

# Load Model
model=keras.models.load_model("")

x=defaultdict(list)
y=defaultdict(list)

for i in raw_data:
    l=len(raw_data[i])
    for j in range(l):
        score=round(raw_data[i][j][1])
        x[score].append(raw_data[i][j][0])
        y[score].append(raw_data[i][j][1])

# Processing images (reshaping, normalizing) before evaluation
for i in range(0,11):
  x[i]=np.array(x[i])
  image_size=x[i].shape[-1]
  train_size=x[i].shape[0]
  x[i]=(x[i].reshape(train_size, image_size, image_size, 1))
  x[i]=x[i]//255
  y[i]=np.array(y[i])

# Evalutaing model for each score class
for i in range(0,11):
    y_pred=model.predict(x[i])
    history = MET(y[i],y_pred)
    a=sum(history)/len(x[i])
    print("Percentage error for score ",i," :",end=" ")
    tf.print(a)
print("---------------------------------------------------")

"""### Checking for individual score class groups

We divide the dataset into three groups, namely: Low (1<=score<4),  Medium (4<=score<7) and High (7<=score<=10)
"""

# Loading Model
model=keras.models.load_model("")

x=defaultdict(list)
y=defaultdict(list)

for i in raw_data:
    l=len(raw_data[i])
    for j in range(l):
        score=round(raw_data[i][j][1])
        if score<4:
          x["low"].append(raw_data[i][j][0])
          y["low"].append(raw_data[i][j][1])
        elif score<7:
          x["medium"].append(raw_data[i][j][0])
          y["medium"].append(raw_data[i][j][1])
        else:
          x["high"].append(raw_data[i][j][0])
          y["high"].append(raw_data[i][j][1])

# Processing images in similar fashion as done in previous methods before evaluation
for i in x:
  x[i]=np.array(x[i])
  image_size=x[i].shape[-1]
  train_size=x[i].shape[0]
  x[i]=(x[i].reshape(train_size, image_size, image_size, 1))
  x[i]=x[i]//255
  y[i]=np.array(y[i])

# Evaluating the model for each group

results=[]
for i in x:
  X=x[i]
  Y=y[i]
  print(X.shape,Y.shape)
  print(len(X))
  history=model.evaluate(X,Y)
  results.append([round(history[1],4),i])
  print(f'Model evaluation for score {i}', history[1])
print("---------------------------------------------------")
print(sorted(results))