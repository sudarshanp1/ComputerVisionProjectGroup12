'''
VERSION NO. = RescueNet_v1.0
##----------------------------------------------
Features: 
	1. Total class: 11 ('Background':0, 'Debris':1, 'Water':2, 'Building_No_Damage':3, 'Building_Minor_Damage':4,
 'Building_Major_Damage':5, 'Building_Total_Destruction':6, 'Vehicle':7, 'Road':8, 'Tree':9, 'Pool':10, 'Sand':11).
	2. Total image: 4494 (Train: 3595, Val: 449, Test: 450)
'''
from tensorflow.keras.utils import Sequence
import os
import cv2
import numpy as np
import tensorflow as tf

class RescuenetDataset(Sequence):
    def __init__(self, image_dir, mask_dir, image_ids, batch_size=8, img_size=(256, 256), num_classes=12):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for img_id in batch_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{img_id}_lab.png")

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

            images.append(img)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)
