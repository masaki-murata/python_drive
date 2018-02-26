# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:14 2018

@author: murata
"""

import numpy as np
from keras.utils.training_utils import multi_gpu_model
import keras


def sensitivity_specificity(path_to_model,
                            crop_size=(64,64),
                            threshold=0.5,
                            batch_size=32,
                            nb_gpus=1,
                            ):
    path_to_validation_data = "../IntermediateData/validation_data.npy"
    path_to_validation_label = "../IntermediateData/validation_label.npy"
    data = np.load(path_to_validation_data)
    label = np.load(path_to_validation_label)
    
    model_single_gpu = keras.models.load_model(path_to_model)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
    
    predicted = model_multi_gpu.predict(data, batch_size=batch_size)
    predicted[predicted>threshold] = 1
    predicted[predicted<=threshold] = 0
    
    sensitivity = predicted[(predicted==1) & (label==1)].size / float(label[label==1].size)
    specificity = predicted[(predicted==0) & (label==0)].size / float(label[label==0].size)

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
