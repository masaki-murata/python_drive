# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:23:25 2018

@author: murata
"""

import numpy as np
from PIL import Image, ImageOps

def load_data():
    path_to_train_images = "../training/images/%d_training.tif"
    for image_id in range(21, 41):
        image = np.array( Image.open(path_to_train_images % image_id) )
        print(image.shape)
        
load_data()