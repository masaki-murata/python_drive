#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:22:59 2018

@author: muratamasaki
"""

import numpy as np
from PIL import Image
import os
import seunet_model, seunet_main
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

def load_image_manual(image_ids=np.arange(18),
                      data_shape=(584,565),
                      crop_shape=(64,64),
                      ):
    path_to_train_image = "../training/images/%d_training.tif" # % image_id
    path_to_train_manual = "../training/1st_manual/%d_manual1.gif" # % image_id

    # load data
    images = np.zeros( (image_ids.shape+data_shape+(3,)), dtype=np.uint8 )
    manuals = np.zeros( (image_ids.shape+data_shape+(1,)), dtype=np.uint8 )
    for x in range(image_ids.size):
        image_id = image_ids[x]
        images[x] = np.array( Image.open(path_to_train_image % (image_id+21)) )
        manual = np.array( Image.open(path_to_train_manual % (image_id+21)) )
        manuals[x] = manual.reshape(manual.shape+(1,))
        
    return images, manuals

def make_validation_dataset(validation_ids=np.arange(18,20),
                            load = True,
                            val_data_size = 2048,
                            data_shape=(584,565),
                            crop_shape=(64,64),
                            ):
    path_to_validation_data = "../IntermediateData/validation_data.npy"
    path_to_validation_label = "../IntermediateData/validation_label.npy"
    if load==True and os.path.exists(path_to_validation_data) and os.path.exists(path_to_validation_label):
        data = np.load(path_to_validation_data)
        label = np.load(path_to_validation_label)
    else:
        images, manuals = load_image_manual(image_ids=validation_ids,
                                            data_shape=data_shape,
                                            crop_shape=crop_shape,
                                            )
        data = np.zeros( (val_data_size,)+crop_shape+(3,), dtype=np.uint8 )
        label = np.zeros( (val_data_size,)+crop_shape+(1,), dtype=np.uint8 )
        for count in range(val_data_size):
            image_id = np.random.randint(images.shape[0])
            y = np.random.randint(images.shape[1]-crop_shape[0])
            x = np.random.randint(images.shape[2]-crop_shape[1])
            data[count] = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
            label[count] = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
                
    return data, label        


def batch_iter(images=np.array([]), # (画像数、584, 565, 3)
               manuals=np.array([]), # (画像数、584, 565, 1)
               crop_shape=(64,64),
               steps_per_epoch=2**14,
#               image_ids=np.arange(20),
               batch_size=32,
               ):
    while True:
        for step in range(steps_per_epoch):
            data = np.zeros( (batch_size,)+crop_shape+(3,), dtype=np.uint8 )
            label = np.zeros( (batch_size,)+crop_shape+(1,), dtype=np.uint8 )
            for count in range(batch_size):
                image_id = np.random.randint(images.shape[0])
                y = np.random.randint(images.shape[1]-crop_shape[0])
                x = np.random.randint(images.shape[2]-crop_shape[1])
                data[count] = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
                label[count] = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
            yield data, label
            

def train(train_ids=np.arange(18),
          validation_ids=np.arange(18,20),
          val_data_size = 2048,
          batch_size=32,
          steps_per_epoch=2**14,
          epochs=256,
          data_shape=(584,565),
          crop_shape=(64,64),
          ):
    
    # set our model
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model = seunet_model.seunet(img_dims, output_dims)
    
    # load data
    train_images, train_manuals = load_image_manual(image_ids=train_ids,data_shape=data_shape,crop_shape=crop_shape)
#    validation_images, validation_manuals = \
#        load_image_manual(image_ids=validation_ids,data_shape=data_shape,crop_shape=crop_shape)
    val_data, val_label = make_validation_dataset(validation_ids=validation_ids,
                                                  load = True,
                                                  val_data_size = 2048,
                                                  data_shape=data_shape,
                                                  crop_shape=crop_shape,
                                                  )
        
    train_gen = batch_iter(images=train_images,
                           manuals=train_manuals, 
                           crop_shape=crop_shape,
                           steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size,
                           )

    path_to_save_model = "../output/ep{epoch:04d}-valloss{val_loss:.4f}.h5"
    callbacks = []
    callbacks.append(ModelCheckpoint(path_to_save_model, monitor='val_loss', save_best_only=False))
#    callbacks.append(CSVLogger("log%03d.csv" % counter))
#    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001 , patience=patience))
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=seunet_main.mean_dice_coef_loss, optimizer=opt_generator)
    
    history = model.fit_generator(train_gen,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
#                                 callbacks=callbacks,
                                 validation_data=(val_data,val_label)
                                 )

def main():
    train(train_ids=np.arange(18),
          validation_ids=np.arange(18,20),
          val_data_size = 2048,
          batch_size=32,
          steps_per_epoch=2**14,
          epochs=256,
          data_shape=(584,565),
          crop_shape=(64,64),
          )    
if __name__ == '__main__':
    main()
    