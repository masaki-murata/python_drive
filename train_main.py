#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:22:59 2018

@author: muratamasaki
"""

import numpy as np
from PIL import Image
import os, datetime
import seunet_model, seunet_main
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2,3", # specify GPU number
        allow_growth=True
    )
)
    
set_session(tf.Session(config=config))

    
def load_image_manual(image_ids=np.arange(20,39),
                      data_shape=(584,565),
#                      crop_shape=(64,64),
                      ):
    path_to_train_image = "../training/images/%d_training.tif" # % image_id
    path_to_train_manual = "../training/1st_manual/%d_manual1.gif" # % image_id

    # load data
    images = np.zeros( (image_ids.shape+data_shape+(3,)), dtype=np.uint8 )
    manuals = np.zeros( (image_ids.shape+data_shape+(1,)), dtype=np.uint8 )
    for x in range(image_ids.size):
        image_id = image_ids[x]
        images[x] = np.array( Image.open(path_to_train_image % (image_id)) )
        manual = np.array( Image.open(path_to_train_manual % (image_id)) )
        manual[manual>0] = 1
        manuals[x] = manual.reshape(manual.shape+(1,))
        
    return images, manuals

def make_validation_dataset(validation_ids=np.arange(39,41),
                            load = True,
                            val_data_size = 2048,
                            data_shape=(584,565),
                            crop_shape=(64,64),
                            ):
    path_to_validation_data = "../IntermediateData/validation_data_crop%d%d.npy" % (crop_shape[0], crop_shape[1])
    path_to_validation_label = "../IntermediateData/validation_label_crop%d%d.npy" % (crop_shape[0], crop_shape[1])
    if load==True and os.path.exists(path_to_validation_data) and os.path.exists(path_to_validation_label):
        data = np.load(path_to_validation_data)
        labels = np.load(path_to_validation_label)
    else:
        images, manuals = load_image_manual(image_ids=validation_ids,
                                            data_shape=data_shape,
#                                            crop_shape=crop_shape,
                                            )
        data = np.zeros( (val_data_size,)+crop_shape+(3,), dtype=np.uint8 )
        labels = np.zeros( (val_data_size,)+crop_shape+(1,), dtype=np.uint8 )
        for count in range(val_data_size):
            image_num = np.random.randint(images.shape[0])
            y = np.random.randint(images.shape[1]-crop_shape[0])
            x = np.random.randint(images.shape[2]-crop_shape[1])
            data[count] = images[image_num, y:y+crop_shape[0], x:x+crop_shape[1],:]
            labels[count] = manuals[image_num, y:y+crop_shape[0], x:x+crop_shape[1],:]
        np.save(path_to_validation_data, data)
        np.save(path_to_validation_label, labels)
                
    return data, labels        


def batch_iter(images=np.array([]), # (画像数、584, 565, 3)
               manuals=np.array([]), # (画像数、584, 565, 1)
               crop_shape=(64,64),
               steps_per_epoch=2**14,
#               image_ids=np.arange(20),
               batch_size=32,
               ):
        
    manuals = manuals.reshape(manuals.shape[:-1])
    while True:
        for step in range(steps_per_epoch):
            data = np.zeros( (batch_size,)+crop_shape+(3,), dtype=np.uint8 )
            labels = np.zeros( (batch_size,)+crop_shape+(1,), dtype=np.uint8 )
            for count in range(batch_size):
                image_num = np.random.randint(images.shape[0])
                theta = np.random.randint(360)
                (h, w) = crop_shape # w は横、h は縦
                c, s = np.abs(np.cos(np.deg2rad(theta))), np.abs(np.sin(np.deg2rad(theta)))
                (H, W) = (int(s*w + c*h), int(c*w + s*h)) #最終的に切り出したい画像に内接する四角形の辺の長さ
                y, x = np.random.randint(images.shape[1] - H + 1), np.random.randint(images.shape[2] - W + 1)
#                y = np.random.randint(images.shape[1]-crop_shape[0])
#                x = np.random.randint(images.shape[2]-crop_shape[1])
#                data[count] = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                label[count] = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                data_crop = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]       
#                label_crop = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
                data_crop, label_crop = Image.fromarray(images[image_num, y:y+H, x:x+W,:]), Image.fromarray(manuals[image_num, y:y+H, x:x+W])
                data_crop, label_crop = np.array(data_crop.rotate(-theta, expand=True)), np.array(label_crop.rotate(-theta, expand=True))
                y_min, x_min = data_crop.shape[0]//2-h//2, data_crop.shape[1]//2-w//2
                data_crop, label_crop = data_crop[y_min:y_min+h, x_min:x_min+w,:], label_crop[y_min:y_min+h, x_min:x_min+w]
                label_crop = label_crop.reshape(label_crop.shape+(1,))
                if np.random.choice([True,False]):
                    data_crop, label_crop = np.flip(data_crop, axis=1), np.flip(label_crop, axis=1)
#                if np.random.choice([True,False]):
#                    data_crop, label_crop = np.flip(data_crop, axis=2), np.flip(label_crop, axis=2)
                data[count], labels[count] = data_crop, label_crop
            yield data, labels
            

def train(train_ids=np.arange(20,38),
          validation_ids=np.arange(18,20),
          val_data_size = 2048,
          batch_size=32,
          data_size_per_epoch=2**14,
#          steps_per_epoch=2**14,
          epochs=256,
          data_shape=(584,565),
          crop_shape=(64,64),
          nb_gpus=1,
          ):
    
    steps_per_epoch=data_size_per_epoch//batch_size
    # set our model
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims)
    print(nb_gpus)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu

    
    # load data
    train_images, train_manuals = load_image_manual(image_ids=train_ids,data_shape=data_shape)
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

#    path_to_save_model = "../output/ep{epoch:04d}-valloss{val_loss:.4f}.h5"
    path_to_cnn_format = "../output/mm%02ddd%02d_%02d/"
    # make a folder to save history and models
    now = datetime.datetime.now()
    for count in range(10):
        path_to_cnn = path_to_cnn_format % (now.month, now.day, count)
        if not os.path.exists(path_to_cnn):
            os.mkdir(path_to_cnn)
            break
    path_to_save_model = path_to_cnn + "model_epoch=%03d.h5"
    path_to_save_weights = path_to_cnn + "weights_epoch=%03d.h5"
    
#    callbacks = []
#    callbacks.append(ModelCheckpoint(path_to_save_model, monitor='val_loss', save_best_only=False))
#    callbacks.append(CSVLogger("log%03d.csv" % counter))
#    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001 , patience=patience))
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    model.compile(loss='binary_crossentropy', optimizer=opt_generator)
    model_multi_gpu.compile(loss=seunet_main.mean_dice_coef_loss, optimizer=opt_generator)
    
    for epoch in range(1,epochs+1):
        model_multi_gpu.fit_generator(train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=1,
#                                      epochs=epochs,
#                                      callbacks=callbacks,
                                      validation_data=(val_data,val_label)
                                      )
        print('Epoch %s/%s done' % (epoch, epochs))
        print("")
        
        if epoch>0 and epoch % 32==0:
            print(epoch)
            model_single_gpu.save(path_to_save_model % (epoch))
            model_single_gpu.save_weights(path_to_save_weights % (epoch))


def main():
    train(train_ids=np.arange(18),
          validation_ids=np.arange(18,20),
          val_data_size = 2048,
          batch_size=32,
          data_size_per_epoch=2**14,
          epochs=256,
          data_shape=(584,565),
          crop_shape=(128,128),
          nb_gpus=1
          )    
if __name__ == '__main__':
    main()
    