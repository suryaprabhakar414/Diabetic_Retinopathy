from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from skimage.io import imread
import os
from keras.utils import np_utils
import cv2
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


traingen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    
valgen=ImageDataGenerator(rescale=1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

seed  =1 

## DATA LOADING

train_set= traingen.flow_from_directory(directory='DR\Train',target_size=(299,299,3),
                                            batch_size=2,class_mode='categorical',seed=seed)
val_set = valgen.flow_from_directory(directory='DR\Val',target_size=(299,299,3),
                                            batch_size=2,class_mode='categorical',seed=seed)
input_tensor=Input(shape=(299,299,3))

## MODEL DEFINITION

def Model_def(pretrained_weights = None):
  base_model = InceptionResNetV2(input_shape =input_tensor,weights = 'imagenet',include_top = False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(5, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['categorical_accuracy'])
  if(pretrained_weights):
     model.load_weights(pretrained_weights)
  return model
  
model = Model_def(pretrained_weights = 'DR/Data/Weights_InceptionResNetV2/InceptionResNetV2.01-1.253-0.1903.hdf5')

## TRAINING 
checkpoint = ModelCheckpoint('DR/Data/Weights_InceptionResNetV2/InceptionResNetV2.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
callbacks_list = [checkpoint,  reduceLROnPlat]
history=model.fit_generator(generator = train_set,steps_per_epoch=1424,epochs=100,callbacks=callbacks_list,
                                    validation_data=val_set,validation_steps= 117,verbose=1)
