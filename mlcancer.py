# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:02:47 2019

@author: JEET
"""

from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt





IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 80000



os.listdir('../hist_cancer_data')



print(len(os.listdir('../hist_cancer_data/Train')))
print(len(os.listdir('../hist_cancer_data/Test')))



df_data = pd.read_csv('../hist_cancer_data/train_labels.csv')
# removing this image because it caused a training error previously
df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
# removing this image because it's black
df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
print(df_data.shape)



df_data['label'].value_counts()



def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):
    
   
    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 
                         figsize=(4*figure_cols,4*len(categories))) # adjust size here
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols) # figure_cols is also the sample size
        for j in range(0,figure_cols):
            file=IMAGE_PATH + sample.iloc[j]['id'] + '.tif'
            im=cv2.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=16)  
    plt.tight_layout()
    plt.show()
    
    
IMAGE_PATH = '../hist_cancer_data/Train/' 

draw_category_images('label',4, df_data, IMAGE_PATH)




#Create train and val set:

df_data.head()



# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
# filter out class 1
df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# concat the dataframes
df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# shuffle
df_data = shuffle(df_data)

df_data['label'].value_counts()



df_data.head()



# train_test_split

# stratify=y creates a balanced validation set.
y = df_data['label']

df_train, df_vald = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_vald.shape)



df_train['label'].value_counts()

df_vald['label'].value_counts()




#Create a Directory Structure
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 2 folders inside 'base_dir':

# train_dir
    # a_no_tumor_tissue
    # b_has_tumor_tissue

# val_dir
    # a_no_tumor_tissue
    # b_has_tumor_tissue



# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)



# [CREATE FOLDERS INSIDE THE TRAIN AND VALIDATION FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# create new folders inside val_dir
no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# check that the folders have been created
os.listdir('base_dir/train_dir')






#Transfer the images into the folders:

# Set the id as the index in df_data
df_data.set_index('id', inplace=True)


# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_vald['id'])



# Transfer the train images

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    # source path to image
    src = os.path.join('../hist_cancer_data/train', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the val images

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    

    # source path to image
    src = os.path.join('../input/train', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)
    
    
    
    
# Build Generators:
    
training_mod = 'base_dir/train_dir'
validate_mod = 'base_dir/val_dir'
test_mod = '../hist_cancer_data/Test'


num_train_img = len(df_train)
num_val_img = len(df_vald)


trn_batch_size=12
val_batch_size=12

train_steps = np.floor(num_train_img / trn_batch_size)
val_steps = np.floor(num_val_img / val_batch_size)

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(training_mod,target_size=(96,96), batch_size=12, class_mode='categorical', seed=30, shuffle=True)

val_gen = datagen.flow_from_directory(validate_mod, target_size=(96,96), batch_size=12, class_mode='categorical', seed=30, shuffle=True)

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(validate_mod, target_size=(96,96),batch_size=3, class_mode='categorical', shuffle=False)
    

    
    
#Model Building:
filter_size = (3,3)
pool_size= (2,2)
filter_1 = 32
filter_2 = 64
filter_3 = 128


dropout_dense = 0.3
dropout_conv = 0.3



#Fix the random seed
seed=7

model = Sequential()


# Applying filter1
model.add(Conv2D(32, (3,3), input_shape = (96, 96, 3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3),padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))




#Applying filter 2:
model.add(Conv2D(64, (3,3), activation ='relu',padding='same'))
model.add(Conv2D(64, (3,3), activation ='relu',padding='same'))
model.add(Conv2D(64, (3,3), activation ='relu',padding='same'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))


#Applying filter 3:
model.add(Conv2D(128, (3,3), activation ='relu',padding='same'))
model.add(Conv2D(128, (3,3), activation ='relu',padding='same'))
model.add(Conv2D(128, (3,3), activation ='relu',padding='same'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()





# Train the Model:
model.compile(loss='binary_crossentropy',optimizer=Adam(0.00024),metrics=['acc'])



#model.compile(optimizer=tf.keras.optimizers.Adam(),
 #               loss=tf.keras.losses.sparse_categorical_crossentropy,
  #              metrics=['acc'])


#model.compile(Adam(lr=0.00024), loss='binary_crossentropy', 
#              metrics=['acc'])

# Get the labels that are associated with each index
print(val_gen.class_indices)





# Build Checkpoints:
checkpoint_path = "train/cp.ckpt"

checkpoint =  tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc', 
                           save_best_only=True , verbose=1)

#checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, 
 #                            save_best_only=True, mode='max')



# To reduce lr
check_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.0012, verbose=1)

                              
#callbacks_list = [checkpoint, check_lr]                         
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=15, verbose=1,
                   callbacks=[checkpoint, check_lr])        


