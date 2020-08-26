# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:02:47 2019

@author: JEET
"""

import os
import cv2
import pandas as pd 
import numpy as np
from numpy.random import seed
seed(101)
import tensorflow as tf
from tensorflow import keras 
from tensorflow import set_random_seed
set_random_seed(101)
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D  
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix
import shutil 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score  
import itertools
from datapreprocessing import *

IMAGE_SIZE = 90  #size of the images 90x90
IMAGE_CHANNELS = 3  #colored rgb images 
SAMPLE_IMAGES = 80000  #taking 80000 images for our model

os.listdir('../9039/PROJECT/histopathologic-cancer-detection')

print(len(os.listdir('../hist_cancer_data/Train')))
print(len(os.listdir('../hist_cancer_data/Test')))
train_labels = pd.read_csv('../hist_cancer_data/train_labels.csv')

#removing images which does not consist of cells 
# removing bright image
train_labels[train_labels['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
# removing black image
train_labels[train_labels['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
print(train_labels.shape)

train_labels['label'].value_counts()

def draw_category_images(col_name,figure_cols, df, TRAIN_PATH):
    
   #adjusting size 
    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 
                         figsize=(4*figure_cols,4*len(categories)))
   
#drawing images for each location
    for i, cat in enumerate(categories):
        
        #figure_cols is sample size 
        sample = df[df[col_name]==cat].sample(figure_cols)
        for j in range(0,figure_cols):
            file=TRAIN_PATH + sample.iloc[j]['id'] + '.tif'
            im=cv2.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=16)  
    plt.tight_layout()
    plt.show()
    
TRAIN_PATH = '../hist_cancer_data/Train/' 

draw_category_images('label',4, train_labels, TRAIN_PATH)
train_labels.head()

#taking random sample of class 0 with size equal to samples in class 1
df_0 = train_labels[train_labels['label'] == 0].sample(SAMPLE_IMAGES, random_state = 101)
#filtering class 1
df_1 = train_labels[train_labels['label'] == 1].sample(SAMPLE_IMAGES, random_state = 101)

train_lables = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
#shuffleing data
train_labels = shuffle(train_labels)
train_labels['label'].value_counts()
train_labels.head()

#spliting data in train and vald
y = train_labels['label']
train, vald = train_test_split(train_labels, test_size=0.10, random_state=101, stratify=y)
print(train.shape)
print(vald.shape)
train['label'].value_counts()
vald['label'].value_counts()

#Create a Directory 
root_dir = 'root_dir'
os.mkdir(root_dir)

#creating two folders inside root_dir
root_train_dir = os.path.join(root_dir, 'root_train_dir')
os.mkdir(root_train_dir)

root_val_dir = os.path.join(root_dir, 'root_val_dir')
os.mkdir(root_val_dir)

#creating sub folders inside root_train_dir for each class
no_tumor_tissue = os.path.join(root_train_dir, 'no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(root_train_dir, 'has_tumor_tissue')
os.mkdir(has_tumor_tissue)

#creating sub folders inside root_val_dir for each class
no_tumor_tissue = os.path.join(root_val_dir, 'no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(root_val_dir, 'has_tumor_tissue')
os.mkdir(has_tumor_tissue)

os.listdir('root_dir/root_train_dir')

#transfering images into folders
train_labels.set_index('id', inplace=True)

train_list = list(train['id'])
vald_list = list(vald['id'])

#transfering train images
for image in train_list:
#adding .tif extension in id csv file  
    fname = image + '.tif'
    target = train_labels.loc[image,'label']
    if target == 0:
        label = 'no_tumor_tissue'
    if target == 1:
        label = 'has_tumor_tissue'
    
    #source path to the image
    src = os.path.join('../hist_cancer_data/train', fname)
    #destination path to the image
    dst = os.path.join(root_train_dir, label, fname)
    shutil.copyfile(src, dst)
    
#transfering vald images
for image in vald_list:
#adding .tif extension in id csv file 
    fname = image + '.tif'
    target = train_labels.loc[image,'label']
    if target == 0:
        label = 'no_tumor_tissue'
    if target == 1:
        label = 'has_tumor_tissue'
    
    #source path to the image
    src = os.path.join('../input/train', fname)
    #destination path to the image
    dst = os.path.join(root_val_dir, label, fname)
    shutil.copyfile(src, dst)

#building generators

training = 'root_dir/root_train_dir'
validate = 'root_dir/root_val_dir'
test = '../hist_cancer_data/Test'
num_train_img = len(train)
num_val_img = len(vald)
train_batch_size=12
val_batch_size=12

train_steps = np.floor(num_train_img / train_batch_size)
val_steps = np.floor(num_val_img / val_batch_size)
datagenerator = ImageDataGenerator(rescale=1.0/255)
train_generator = datagenerator.flow_from_directory(training,target_size=(90,90), batch_size=12, class_mode='categorical', seed=30, shuffle=True)
val_generator = datagenerator.flow_from_directory(validate, target_size=(90,90), batch_size=12, class_mode='categorical', seed=30, shuffle=True)
#as do not want to shuffle test data shuffle=false 
test_generator = datagenerator.flow_from_directory(validate, target_size=(90,90),batch_size=3, class_mode='categorical', shuffle=False)


#CNN model building

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
#Applying filter1
model.add(Conv2D(32, (3,3), input_shape = (90, 90, 3),padding='same'))
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
#Train the Model:
model.compile(loss='binary_crossentropy',optimizer=Adam(0.00024),metrics=['acc'])
#model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.sparse_categorical_crossentropy,
  #              metrics=['acc'])
#model.compile(Adam(lr=0.00024), loss='binary_crossentropy', 
#              metrics=['acc'])
# Get the labels that are associated with each index
print(val_generator.class_indices)
# Build Checkpoints:
checkpoint_path = "train/cp.ckpt"
checkpoint =  tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc', 
                           save_best_only=True , verbose=1)
#checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, 
#                            save_best_only=True, mode='max')
# To reduce lr
check_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.0012, verbose=1)
                              
#callbacks_list = [checkpoint, check_lr]                         
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, 
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=15, verbose=1,
                   callbacks=[checkpoint, check_lr])   
model.metrics_names

#here the best epoch is used
model.load_weights('his_cancer.h5')

validation_loss, validation_acc = \
model.evaluate_generator(test_generator, 
                        steps=len(vald)) 

print('validation_loss:', validation_loss)
print('validation_acc:', validation_acc)

#displaying loss and accuracy curves 

acc = history.history['acc']
validation_acc = history.history['validation_acc']
loss = history.history['loss']
validation_loss = history.history['validation_loss']

epochs = range(1, len(acc) + 1)

#ploting loss 
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

#ploting accuracy 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, validation_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

#prediction on the validation set
predictions = model.predict_generator(test_generator, steps=len(vald), verbose=1)
predictions.shape 
test_generator.class_indices


df_prediction = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

df_prediction.head()

#true labels
y_true = test_generator.classes

#predicted labels as probabilities
y_pred = df_prediction['has_tumor_tissue']

#AUC score
roc_auc_score(y_true, y_pred)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues): 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion_matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")
#ploting confusion matrix
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
#labels of the test images
test_labels = test_generator.classes
test_labels.shape

#argmax is used to return the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_generator.class_indices

#define the labels
cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


#creating test directory 
root_test_dir = 'root_test_dir'
os.mkdir(root_test_dir)
    
#creating test_images inside root_test_dir folder 
test_images = os.path.join(root_test_dir, 'test_images')
os.mkdir(test_images)

os.listdir('root_test_dir')

test_list = os.listdir('../histopathologic-cancer-detection/test')

for image in test_list:
    
    fname = image
    #source path to the image
    src = os.path.join('../histopathologic-cancer-detection/test', fname)
    #destination path to the image
    dst = os.path.join(test_images, fname)
    shutil.copyfile(src, dst)
len(os.listdir('test_dir/test_images'))
test_path ='test_dir'

#here change the path to point to the test_images folder
test_generator = datagenerator.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
#taking 57458 images from test
num_test_images = 57458

#using the best epoch
model.load_weights('his_cancer.h5')

predictions = model.predict_generator(test_generator, steps=num_test_images, verbose=1)
len(predictions)
df_prediction = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])
df_prediction.head()

test_filenames = test_generator.filenames

#adding filenames to the dataframe
df_prediction['file_names'] = test_filenames
df_prediction.head()

#creating an id 
def extract_id(x):
#spliting into a list
    a = x.split('/')
    b = a[1].split('.')
    extracted_id = b[0]
    return extracted_id

df_prediction['id'] = df_prediction['file_names'].apply(extract_id)
df_prediction.head()
#predict probability that image has tumor tissue 
y_pred = df_prediction['has_tumor_tissue']

# get the id column
image_id = df_prediction['id']
       


