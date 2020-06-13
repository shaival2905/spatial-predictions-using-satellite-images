#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:34:22 2020

@author: shaivalshah
"""

from preprocessing import Preprocessing
from model import UnetModel

data_path = '..'
"""
To train different classes
for buildings: class_dict = {1: 0}
for road:      class_dict = {3: 0}
for tracks:    class_dict = {4, 0}
"""
class_dict = {1:0}
Patch_size = 224
N_split = 15
n_classes = len(class_dict)
inp_shape = (Patch_size, Patch_size, 20)
preprocessor = Preprocessing(data_path, class_dict)
unet_model = UnetModel(inp_shape, n_classes)
unet_model.getModel(0.2)
print ("Model generated")

unet_model.compileModel()
print ("Model compiled")
epochs = 100
batch_size = 16

train_image_data_gen = preprocessor.imagePatchGenerator(batch_size)
val_image_data_gen = preprocessor.imagePatchGenerator(batch_size, val_data = True)
print("batch generators generated")

print("Training...")
trained_model = unet_model.train_generator(batch_size, 
                                           epochs,
                                           train_image_data_gen,
                                           val_image_data_gen)
print("model trained")



import numpy as np

