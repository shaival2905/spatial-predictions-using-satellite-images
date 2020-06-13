#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:46:58 2020

@author: shaivalshah
"""

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras.backend as K
import numpy as np
import os

class UnetModel:
    
    def __init__(self,
                 input_shape,
                 n_classes,
                 filter_seq = [32, 64, 128, 256, 512, 1024],
                 dropout_seq = [None, None, None, None, 0.5, 0.5],
                 kernel_size = 3,
                 activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal',
                 patience = 3):
        assert len(filter_seq) > 0
        assert len(filter_seq) >= len(dropout_seq)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_classes = n_classes
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.filter_seq = filter_seq
        self.dropout_seq = dropout_seq
        if len(self.filter_seq) > len(self.dropout_seq):
            pad = len(self.filter_seq) - len(self.dropout_seq)
            pad_arr = [None]*pad
            self.dropout_seq += pad_arr
        if not os.path.isdir("weights"):
            os.mkdir("weights")
    
    def getModel(self):
        
        def encoderBlock(inp, 
                         filters,
                         kernel_size,
                         block_num,
                         strides = 1,
                         activation = 'relu',
                         padding = 'same',
                         kernel_initializer = 'he_normal',
                         dropout_rate = None):
            
            conv = Conv2D(filters,
                          kernel_size,
                          strides = strides,
                          activation = activation,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          name = 'encoder_layer_{}_1'.format(block_num))(inp)
            conv = Conv2D(filters,
                          kernel_size,
                          strides = strides,
                          activation = activation,
                          padding = padding, 
                          kernel_initializer = kernel_initializer,
                          name = 'encoder_layer_{}_2'.format(block_num))(conv)
            if dropout_rate:
                conv = Dropout(dropout_rate, name = 'enc_dropout_layer_{}'.format(block_num))(conv)
            out = MaxPooling2D(pool_size=(2, 2), name = 'enc_maxpool_layer_{}'.format(block_num))(conv)
            return conv, out
        
        def decoderBlock(inp, 
                         filters,
                         kernel_size,
                         conv_block,
                         block_num,
                         strides = 1,
                         activation = 'relu',
                         padding = 'same',
                         kernel_initializer = 'he_normal',
                         dropout_rate = None):
            
            conv = UpSampling2D(size = (2,2))(inp)
            conv = Conv2D(filters,
                          2,
                          activation = activation,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          name = 'deconvolve_layer_{}'.format(block_num))(conv)
            conv = concatenate([conv_block, conv], axis = 3)
            if block_num == len(self.filter_seq):
                filters *= 2
            conv = Conv2D(filters,
                          kernel_size,
                          activation = activation,
                          padding = padding,
                          strides = strides,
                          kernel_initializer = kernel_initializer,
                          name = 'decoder_layer_{}_1'.format(block_num))(conv)
            conv_out = Conv2D(filters,
                          kernel_size,
                          activation = activation,
                          padding = padding,
                          strides = strides,
                          kernel_initializer = kernel_initializer,
                          name = 'decoder_layer_{}_2'.format(block_num))(conv)
            if dropout_rate:
                conv_out = Dropout(dropout_rate, name = 'dec_dropout_layer_{}'.format(block_num))(conv_out)
            return conv_out
        
        
        input_layer = Input(shape = self.input_shape, name = 'main_input_layer')

        self.latent_layer_params = (self.filter_seq[-1], self.dropout_seq[-1])
        self.filter_seq = self.filter_seq[:-1]
        self.dropout_seq = self.dropout_seq[:-1]
        
        # Adding encoder layers
        encoder_blocks_outs = []
        encoder_blocks_conv = []
        
        for i, (num_filters, dropout_rate)  in enumerate(zip(self.filter_seq, self.dropout_seq)):
            if i == 0:
                encoder_conv, encoder_out = encoderBlock(input_layer,
                                                         num_filters,
                                                         self.kernel_size,
                                                         i+1,
                                                         activation = self.activation,
                                                         padding = self.padding,
                                                         kernel_initializer = self.kernel_initializer,
                                                         dropout_rate = dropout_rate)
            else:
                encoder_conv, encoder_out = encoderBlock(encoder_blocks_outs[i-1],
                                                         num_filters,
                                                         self.kernel_size,
                                                         i+1,
                                                         activation = self.activation,
                                                         padding = self.padding,
                                                         kernel_initializer = self.kernel_initializer,
                                                         dropout_rate = dropout_rate)
            encoder_blocks_outs.append(encoder_out)
            encoder_blocks_conv.append(encoder_conv)
            
        # Adding latent layer
        latent_layer = Conv2D(self.latent_layer_params[0], 
                              self.kernel_size, 
                              activation='relu', 
                              padding='same', 
                              kernel_initializer='he_normal',
                              name = 'latent_layer_1')(encoder_blocks_outs[-1])
        latent_layer_out = Conv2D(self.latent_layer_params[0],
                              self.kernel_size, 
                              activation='relu', 
                              padding='same', 
                              kernel_initializer='he_normal',
                              name = 'latent_layer_2')(latent_layer)
        if self.latent_layer_params[1]:
            latent_layer_out = Dropout(self.latent_layer_params[1], 
                                       name = 'latent_dropout_layer')(latent_layer_out)
        
        decoder_blocks_outs = []
        
        for i, (num_filters, dropout_rate)  in enumerate(zip(reversed(self.filter_seq), reversed(self.dropout_seq))):
            if i == 0:
                decoder_out = decoderBlock(latent_layer_out,
                                           num_filters,
                                           self.kernel_size,
                                           encoder_blocks_conv[-1-i],
                                           i+1,
                                           activation = self.activation,
                                           padding = self.padding,
                                           kernel_initializer = self.kernel_initializer,
                                           dropout_rate = dropout_rate)
            else:
                decoder_out = decoderBlock(decoder_blocks_outs[i-1],
                                           num_filters,
                                           self.kernel_size,
                                           encoder_blocks_conv[-1-i],
                                           i+1,
                                           activation = self.activation,
                                           padding = self.padding,
                                           kernel_initializer = self.kernel_initializer,
                                           dropout_rate = dropout_rate)
            decoder_blocks_outs.append(decoder_out)
        
        
        # Output softmax layer
        softmax_out = Conv2D(self.n_classes, 
                             1, 
                             activation = 'sigmoid' if self.n_classes == 1 else 'softmax',
                             kernel_initializer = self.kernel_initializer,
                             name = 'softmax_layer')(decoder_blocks_outs[-1])        
        
        # Final model
        self.model = Model(inputs=input_layer, outputs=softmax_out)
        
        return self.model
    
    
    def weighted_binary_crossentropy(self, y_true, y_pred):
        wmap = y_true[:, :, :, self.n_classes:]
        y_true = y_true[:, :, :, :self.n_classes]
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        weighted_loss = loss * wmap
        return K.mean(weighted_loss, axis=-1)
    
    def weighted_categorical_crossentropy(self, y_true, y_pred):
        wmap = y_true[:, :, :, self.n_classes:]
        y_true = y_true[:, :, :, :self.n_classes]
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)
        loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
        weighted_loss = loss * wmap
        return K.mean(weighted_loss, axis=-1)
    
    def jaccard_distance(self, y_true, y_pred, smooth=100):
        y_true = y_true[:, :, :, :self.n_classes]
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    
    def jaccard_coef(self, y_true, y_pred):
        # __author__ = Vladimir Iglovikov
        smooth = 1e-12
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return K.mean(jac)


    def jaccard_coef_int(self, y_true, y_pred):
        # __author__ = Vladimir Iglovikov
        smooth = 1e-12
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return K.mean(jac)
    
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-4
        y_true = y_true[:, :, :, :self.n_classes]
        flatten_y_true = K.flatten(y_true)
        flatten_y_pred = K.flatten(y_pred)
        intersection = K.sum(flatten_y_true * flatten_y_pred)
        union = K.sum(flatten_y_true) + K.sum(flatten_y_pred)
        diceCoeff = 2 * (intersection + smooth) / (union + smooth)
        return diceCoeff
    
    def compileModel(self, optimizer = 'adam', loss=None, metrics=None):
        if not metrics:
            metrics=[self.jaccard_coef, self.jaccard_coef_int, self.dice_coef]
        if not loss:
            if self.n_classes > 1:
                loss = self.weighted_categorical_crossentropy
            else:
                loss = self.weighted_binary_crossentropy
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model
    
    def train_generator(self, 
              batch_size,
              epochs,
              train_image_data_gen,
              val_image_data_gen):
        checkpointer = ModelCheckpoint("weights/unet_weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
        self.model.fit_generator(generator = train_image_data_gen,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=val_image_data_gen,
                                 validation_steps=20,
                                 steps_per_epoch=400,
                                 use_multiprocessing=True,
                                 workers=5,
                                 max_queue_size=1,
                                 callbacks=[checkpointer, early_stopping, reduce_lr])
    
    def train(self, 
              train_data_path,
              batch_size,
              epochs,
              val_data_path = None):
        
        data_train = np.load(train_data_path, mmap_mode="r")
        img_train = data_train[:, :, :, :20]
        y_train = np.array(data_train[:, :, :, 20:])
        mean = np.mean(img_train)
        std = np.std(img_train)
        img_train = (img_train - mean)/std
         
        if val_data_path:
            data_val = np.load(val_data_path, mmap_mode="r")
            img_val = data_val[:, :, :, :20]
            y_val = data_val[:, :, :, 20:]
        
        checkpointer = ModelCheckpoint("weights/unet_weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        
        if val_data_path:
            self.model.fit(x = img_train,
                           y = y_train,
                           epochs = epochs,
                           batch_size = batch_size,
                           validation_data = [img_val, y_val],
                           callbacks = [checkpointer, early_stopping, reduce_lr])
        else:
            self.model.fit(x = img_train,
                       y = y_train,
                       epochs = epochs,
                       batch_size = batch_size,
                       validation_split = 0.1,
                       callbacks = [checkpointer, early_stopping, reduce_lr])
    
    # predict function only for binary class prediction
    def predict(self, input_data, thres = 0.5):
        masks = self.model.predict(input_data)
        for i, mask in enumerate(masks):
            masks[i] = np.uint8(mask > thres)
        return masks