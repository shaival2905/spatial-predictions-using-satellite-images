#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:17:45 2020

@author: shaivalshah
"""

import numpy as np
import pandas as pd
import csv
import sys
import cv2
import shapely.wkt
import shapely.affinity
import tifffile as tiff
import os
from skimage.segmentation import find_boundaries

csv.field_size_limit(sys.maxsize)

class Preprocessing:
    
    # Initialising the required parameters for preprocessing
    def __init__(self, data_path, class_dict, w0 = 10, sigma = 5, N_split = 15, Patch_size = 224):
        self.w0 = w0
        self.sigma = sigma
        self.class_dict = class_dict
        self.N_split = N_split
        self.Patch_size = Patch_size
        self.data_path = data_path
        self.grid_sizes_df = pd.read_csv(os.path.join(self.data_path, 'grid_sizes.csv'))
        self.grid_sizes_df = self.grid_sizes_df.rename(columns = {self.grid_sizes_df.columns[0]: 'IM_ID'})
        self.polygons_df = pd.read_csv(os.path.join(self.data_path, 'train_wkt_v4.csv'))
        self.show_mask = lambda mask: tiff.imshow(255*np.stack([mask, mask, mask]))
        
    #Get XMax, YMin to do scaling for each specific image
    def getGridSize(self, img_id):
        return tuple(self.grid_sizes_df[self.grid_sizes_df['IM_ID'] == img_id][['Xmax' ,'Ymin']].values[0])
    
    #Get the polygon for specific image id
    def getPolygons(self, img_id):
        poly_dict = dict()
        img_poly_df = self.polygons_df[self.polygons_df['ImageId'] == img_id]
        classes = img_poly_df['ClassType'].unique()
        for class_type in classes:
            poly_dict[int(class_type)] = shapely.wkt.loads(img_poly_df[img_poly_df['ClassType'] == class_type]['MultipolygonWKT'].values[0])
        return poly_dict
    
    #Read image of selected band where img_band can be 3 band or 16 band
    def getImage(self, img_id, img_band):
        
        if img_band =='RGB':
            file = os.path.join(self.data_path,'three_band', '{}.tif'.format(img_id))
        else:
            file = os.path.join(self.data_path,'sixteen_band', '{}_{}.tif'.format(img_id, img_band))    
        img = tiff.imread(file)
        img = img[:,:,None] if img_band == 'P' else np.rollaxis(img, 0, 3)
        img = img.astype(np.float32)/16384 if img_band == 'A' else img.astype(np.float32)/2048
        return img
    
    #Resize and join images of all 20 bands
    def getImageAllband(self, img_id, Scale_Size = None):
        if not Scale_Size:
            Scale_Size = self.Patch_size * self.N_split
        img_RGB = cv2.resize(self.getImage(img_id, 'RGB'), (Scale_Size, Scale_Size))    
        img_M = cv2.resize(self.getImage(img_id, 'M'), (Scale_Size, Scale_Size))
        img_A = cv2.resize(self.getImage(img_id, 'A'), (Scale_Size, Scale_Size))
        img_P = cv2.resize(self.getImage(img_id, 'P'),( Scale_Size, Scale_Size))
        img = np.concatenate((img_RGB,img_M, img_A,img_P[:,:,None]), axis=2)
        return img
    
    #Generate masks for the selected image
    def getMask(self, img_id, img_shape):
        polygons_vals_all_classes = self.getPolygons(img_id)
        Xmax, Ymin = self.getGridSize(img_id)
        H, W = img_shape
        xfact = W * (W / (W + 1)) / Xmax
        yfact = H * (H / (H + 1)) / Ymin
        total_classes = len(self.class_dict)
        masks = np.zeros((H, W, total_classes), dtype=np.uint8)
        for class_type, polygons_vals in polygons_vals_all_classes.items():
            if class_type in self.class_dict:
                class_mask = np.zeros((H, W), dtype=np.uint8)
                polygon_img = shapely.affinity.scale(polygons_vals, xfact = xfact, yfact = yfact, origin = (0, 0, 0))
                if not polygon_img:
                    continue
                external_region = []
                internal_region = []
                for poly_val in polygon_img:
                    external_region.append(np.array(poly_val.exterior.coords).round().astype(np.int32))
                    for poly_int in poly_val.interiors:
                        internal_region.append(np.array(poly_int.coords).round().astype(np.int32))
                cv2.fillPoly(class_mask, external_region, 1)
                cv2.fillPoly(class_mask, internal_region, 0)
                masks[:, :, self.class_dict[class_type]] = class_mask
        return masks
    
    #Generate WeightMap for corresponding masks
    def getWeightMap(self, masks):
        #__author__ jaidev.github.io
        nrows, ncols = masks.shape[1:]
        dist_mat = np.zeros((nrows * ncols, masks.shape[0]))
        X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
        X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
        for i, mask in enumerate(masks):
            X2, Y2 = np.nonzero(find_boundaries(mask, mode='inner'))
            sum_x = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
            sum_y = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
            if len(sum_x) > 0 and len(sum_y) > 0:
                dist_mat[:, i] = np.sqrt(sum_x + sum_y).min(axis=0)
        ix = np.arange(dist_mat.shape[0])
        if dist_mat.shape[1] == 1:
            d1 = dist_mat.ravel()
            border_loss_map = self.w0 * np.exp((-1 * (d1) ** 2) / (2 * (self.sigma ** 2)))
        else:
            if dist_mat.shape[1] == 2:
                d1_ix, d2_ix = np.argpartition(dist_mat, 1, axis=1)[:, :2].T
            else:
                d1_ix, d2_ix = np.argpartition(dist_mat, 2, axis=1)[:, :2].T
            d1 = dist_mat[ix, d1_ix]
            d2 = dist_mat[ix, d2_ix]
            border_loss_map = self.w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (self.sigma ** 2)))
        xBLoss = np.zeros((nrows, ncols))
        xBLoss[X1, Y1] = border_loss_map
        loss = np.zeros((nrows, ncols))
        w_1 = 1 - masks.sum() / loss.size
        w_0 = 1 - w_1
        loss[masks.sum(0) == 1] = w_1
        loss[masks.sum(0) == 0] = w_0
        ZZ = xBLoss + loss
        return ZZ
    
    """
    Since the original image is too large, we're generating patches from the
    original images with 224 as the input size of the CNN. This value was chosen
    after experimentation to enhance performance.
    
    """
    def getPatch(self, img_id):
        self.N_patch = self.N_split**2
        patch_all = []
        img = self.getImageAllband(img_id)
        masks = self.getMask(img_id, (img.shape[0], img.shape[1]))
        for i in range(self.N_split):
            for j in range(self.N_split):
                y = masks[self.Patch_size*i:self.Patch_size*(i + 1), self.Patch_size*j:self.Patch_size*(j + 1)]
                weight_map_y = []
                for mask in y.transpose(2,0,1):
                    weight_map_y.append(self.getWeightMap(np.array([mask])))
                weight_map_y = np.array(weight_map_y).transpose(1, 2, 0)#self.getWeightMap(y.transpose(2,0,1))
                if np.sum(y) > 0:
                    x = img[self.Patch_size*i:self.Patch_size*(i + 1), self.Patch_size*j:self.Patch_size*(j + 1),:]
                    x = self.normalizedImages(x)
                    patch_all.append(np.concatenate((x, y, weight_map_y), axis = 2))
        patch_all = np.asarray(patch_all)
        return patch_all
    
    #Calls getPatch for all images
    def getAllPatches(self, test_split = 0.2):
        count = 0
        x = []
        #img_ids=["6120_2_2"]
        img_ids = sorted(self.grid_sizes_df.IM_ID.unique())
        for i, img_id in enumerate(img_ids):
            print("Saving for image {}".format(img_id))
            x_all = self.getPatch(img_id)
            if len(x_all) > 0:
                count = count + 1
                if count == 1:
                    x = x_all
                else:
                    x = np.concatenate((x, x_all), axis = 0)
        trn = 1 - test_split
        l = len(x)
        train_stump = int(l * trn)
        val_stump = train_stump + (l - train_stump)//2
        
        np.save('data_pos_%d_%d_train' % (self.Patch_size, self.N_split), x[:train_stump])
        np.save('data_pos_%d_%d_val' % (self.Patch_size, self.N_split), x[train_stump:val_stump])
        np.save('data_pos_%d_%d_test' % (self.Patch_size, self.N_split), x[val_stump:])
        return train_stump, (l - train_stump)//2
    
    #Normalizes the patches.
    def normalizedImages(self, patch_img_batch):
        return (patch_img_batch - np.mean(patch_img_batch))/np.std(patch_img_batch)
    
    #Train using Data Generator. It yields patches as per the batch size.
    def getPatchBatch(self, img_id, batch_size):
        self.N_patch = self.N_split**2
        patch_img_batch = []
        patch_mask_batch = []
        patch_wmap_batch = []
        img = self.getImageAllband(img_id)
        masks = self.getMask(img_id, (img.shape[0], img.shape[1]))
        for i in range(self.N_split):
            for j in range(self.N_split):   
                y = masks[self.Patch_size*i:self.Patch_size*(i + 1), self.Patch_size*j:self.Patch_size*(j + 1)]
                weight_map_y = []
                for mask in y.transpose(2,0,1):
                    weight_map_y.append(self.getWeightMap(np.array([mask])))
                weight_map_y = np.array(weight_map_y).transpose(1, 2, 0)
                if np.sum(y) > 0:
                    x = img[self.Patch_size*i:self.Patch_size*(i + 1), self.Patch_size*j:self.Patch_size*(j + 1),:]
                    patch_img_batch.append(x)
                    patch_mask_batch.append(y)
                    patch_wmap_batch.append(weight_map_y)
                    if len(patch_img_batch) == batch_size:
                        norm_patch_img_batch = self.normalizedImages(np.array(patch_img_batch))
                        yield norm_patch_img_batch, np.array(patch_wmap_batch), np.array(patch_mask_batch)
                        patch_img_batch = []
                        patch_mask_batch = []
                        patch_wmap_batch = []
    
    #Iterates getPatchBatch for every image
    def imagePatchGenerator(self, batch_size, val_data = False, val_split = 0.2):
        while True:            
            img_ids = sorted(self.grid_sizes_df.IM_ID.unique())
            l = len(img_ids)
            if val_data:
                img_ids = img_ids[-int(l*val_split):]
            else:
                img_ids = img_ids[:-int(l*val_split)]
            for img_id in img_ids:
                patch_gen = self.getPatchBatch(img_id, batch_size)
                for X_batch, wmap_batch, mask_batch in patch_gen:
                    y_batch = np.stack([mask_batch, wmap_batch], axis = 3)
                    yield X_batch, y_batch