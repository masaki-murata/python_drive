# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:14 2018

@author: murata
"""

import numpy as np
from keras.utils.training_utils import multi_gpu_model
import keras
import seunet_model, train_main
from scipy.ndimage import label


def sensitivity_specificity(path_to_model_weights,
                            crop_shape=(64,64),
                            threshold=0.5,
                            batch_size=32,
                            nb_gpus=1,
                            ):
    path_to_validation_data = "../IntermediateData/validation_data.npy"
    path_to_validation_label = "../IntermediateData/validation_label.npy"
    data = np.load(path_to_validation_data)
    labels = np.load(path_to_validation_label)
    
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims)
    model_single_gpu.load_weights(path_to_model_weights)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
    
    predicted = model_multi_gpu.predict(data, batch_size=batch_size)
    predicted[predicted>threshold] = 1
    predicted[predicted<=threshold] = 0
    
    sensitivity = predicted[(predicted==1) & (labels==1)].size / float(labels[labels==1].size)
    specificity = predicted[(predicted==0) & (labels==0)].size / float(labels[labels==0].size)

    return sensitivity, specificity


def object_level_dice_2d(y_true, y_pred): # y_true.shape = (画像のindex a, y, x)
    def dice_coeff(g ,s):
        return 2*(np.sum(g*s)+1) / (np.sum(g)+np.sum(s)+1)
    
    s_sum, g_tilde_sum = np.sum(y_pred), np.sum(y_true) # omega, omega_tilde の分子
    
    dice_object=0
    for a in range(len(y_true)):
        labeled_true, num_labels_true = label(y_true[a])
        labeled_pred, num_labels_pred = label(y_pred[a])
        
        # initialize
        g_tilde = np.zeros( (num_labels_true,)+y_true.shape[1:], dtype=np.uint8 )
        s = np.zeros( (num_labels_pred,)+y_true.shape[1:], dtype=np.uint8 )
        omega = np.zeros(num_labels_pred, dtype=np.uint8)
        omega_tilde = np.zeros(num_labels_true, dtype=np.uint8)
        # set g_tilde and s
        for i in range(num_labels_true):
            g_tilde[i][labeled_true==i+1] = 1
            omega_tilde[i] = np.sum(g_tilde[i]) / g_tilde_sum
        for i in range(num_labels_pred):
            s[i][labeled_pred==i+1] = 1
            omega[i] = np.sum(s[i]) / s_sum
        
        # compute Dice(G, S)
        dice_sg = np.zeros(num_labels_pred, dtype=np.uint8)
        for i in range(num_labels_pred):
            dice_sg[i] = 0
            for j in range(num_labels_true):
                dice_sg[i] = max( dice_sg[i], dice_coeff(g_tilde[j], s[i]) )
        # compute Dice(G_tilde, S_tilde)
        dice_sg_tilde = np.zeros(num_labels_true, dtype=np.uint8)
        for i in range(num_labels_true):
            dice_sg_tilde[i] = 0
            for j in range(num_labels_pred):
                dice_sg_tilde[i] = max( dice_sg_tilde[i], dice_coeff(g_tilde[i], s[j]) )
        
        dice_object += 0.5 * ( np.sum(omega*dice_sg) + np.sum(omega_tilde*dice_sg_tilde) )
    
    return dice_object


def whole_slide_dice_coeff(path_to_model_weights,
                           image_ids=np.arange(18,20),
                           data_shape=(584,565),
                           crop_shape=(64,64),
                           nb_gpus=1,
                           ):
    def dice_coeff(g ,s):
        return 2*(np.sum(g*s)+1) / (np.sum(g)+np.sum(s)+1)

    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims)
    model_single_gpu.load_weights(path_to_model_weights)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
        
    images, manuals = train_main.load_image_manual(image_ids=image_ids,
                                                   data_shape=data_shape,
                                                   )
    def dice_coeff_wsi(image_id):
        count = 0
        data_size = (1+data_shape[0]//crop_shape[0]) * (1+data_shape[1]//crop_shape[1])
        data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
        for y in range(0, data_shape[0], crop_shape[0]):
            for x in range(0, data_shape[1], crop_shape[1]):
                data[count] = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
        predicted = model_multi_gpu.predict(data, batch_size=32)
        return dice_coeff(predicted, manuals[image_id])
    
    dcw=0
    for image_id in image_ids:
        dcw += dice_coeff_wsi(image_id)
    
    return dcw / len(image_ids)
    
#        data = np.zeros( (val_data_size,)+crop_shape+(3,), dtype=np.uint8 )
#        labels = np.zeros( (val_data_size,)+crop_shape+(1,), dtype=np.uint8 )
    


def main():
    path_to_model_weights = "../output/mm02dd26_01/weights_epoch=32.h5"
    whole_slide_dice_coeff(path_to_model_weights,
                           image_ids=np.arange(18,20),
                           data_shape=(584,565),
                           crop_shape=(64,64),
                           nb_gpus=1,
                           )
#    sensitivity, specificity = sensitivity_specificity(path_to_model_weights,
#                                                       crop_shape=(64,64),
#                                                       threshold=0.5,
#                                                       batch_size=32,
#                                                       nb_gpus=1,
#                                                       )    
    print(sensitivity, specificity)
    
    
if __name__ == '__main__':
    main()
