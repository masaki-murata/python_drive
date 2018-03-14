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
from PIL import Image


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
#    def dice_coeff(g ,s):
#        return 2*(np.sum(g*s)+1) / (np.sum(g)+np.sum(s)+1)
    def load_model(path_to_model_weights):
        img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
        model_single_gpu = seunet_model.seunet(img_dims, output_dims)
        model_single_gpu.load_weights(path_to_model_weights)
        if int(nb_gpus) > 1:
            model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
        else:
            model_multi_gpu = model_single_gpu
        return model_multi_gpu
    
    model_multi_gpu = load_model(path_to_model_weights)
    
    images, manuals = train_main.load_image_manual(image_ids=image_ids,
                                                   data_shape=data_shape,
                                                   )
    def dice_coeff_wsi(count_image):
        count = 0
        data_size = (1+data_shape[0]//crop_shape[0]) * (1+data_shape[1]//crop_shape[1])
        data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
        for y in range(0, data_shape[0], crop_shape[0]):
            for x in range(0, data_shape[1], crop_shape[1]):
                data[count] = images[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
                count += 1
        predicted = np.round( model_multi_gpu.predict(data, batch_size=32) )
        sum_groundtruth = np.sum(predicted)
        sum_predict = np.sum(images[count_image])
        dice_numerator = 0
        for count in range(data_size):
            dice_numerator += 2 * np.sum( np.data[count] * predicted[count] )
        return dice_numerator / (sum_groundtruth+sum_predict)
    
    dice_sum=0
    for count_image in range(len(image_ids)):
        dice_sum += dice_coeff_wsi(count_image)
    
    return dice_sum / len(image_ids)
    

def whole_slide_accuracy(path_to_model_weights,
                         image_ids=np.arange(39,41),
                         data_shape=(584,565),
                         crop_shape=(64,64),
                         nb_gpus=1,
                         ):
    path_to_mask = "../training/mask/%d_training_mask.gif" # % image_id
    
    def load_model(path_to_model_weights):
        img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
        model_single_gpu = seunet_model.seunet(img_dims, output_dims)
        model_single_gpu.load_weights(path_to_model_weights)
        if int(nb_gpus) > 1:
            model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
        else:
            model_multi_gpu = model_single_gpu
        return model_multi_gpu
    
    model_multi_gpu = load_model(path_to_model_weights)
    
    images, manuals = train_main.load_image_manual(image_ids=image_ids,
                                                   data_shape=data_shape,
                                                   )
    
    pixel_sum, true_sum = 0,0
    mask = np.zeros( (image_ids.shape+data_shape+(1,)), dtype=np.uint8 )
    for count_image in range(image_ids.size):
        image_id = image_ids[count_image]
        mask[count_image] = np.array( Image.open(path_to_mask % (image_id)) )
        
        data_size = (1+data_shape[0]//crop_shape[0]) * (1+data_shape[1]//crop_shape[1])
        data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
        count = 0
        for y in range(0, data_shape[0], crop_shape[0]):
            for x in range(0, data_shape[1], crop_shape[1]):
                data[count] = images[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
                count += 1
        predicted = np.round( model_multi_gpu.predict(data, batch_size=32) )
        pixel_sum += np.sum(manuals[x][mask>0])
        true_sum += np.sum(predicted[mask>0 & manuals[x]==predicted])
    
    return true_sum / float(pixel_sum)
        
    
    

def main():
    path_to_model_weights = "../output/mm03dd02_03/weights_epoch=064.h5"
    whole_slide_dice_coeff(path_to_model_weights,
#                           image_ids=np.arange(18,20),
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
#    print(sensitivity, specificity)
    
    
if __name__ == '__main__':
    main()
