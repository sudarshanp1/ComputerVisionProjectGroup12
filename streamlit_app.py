import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import segmentation_models as sm
import matplotlib.pyplot as plt
import io
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score  # IoU
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,\
                             Dropout, Layer, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow.image as tfimg
import math
import random

import gdown

def download_model(link,name):
    url = link
    output = name
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

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

# Image folder
IMAGE_FOLDER = "images/test-org-img"  # Change to your image folder path

def categorical_encoding_to_prob(img, num_of_classes):
    res_img = np.zeros((img.shape[0], img.shape[1], num_of_classes))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if len(img.shape) == 2:
                index = img[i][j]
            else:
                index = img[i][j][0]
            res_img[i][j][index] = 1

    return res_img

def rgb_to_categorical(img, colors):
    res_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if tuple(img[i][j]) in colors:
                res_img[i][j] = colors.index(tuple(img[i][j]))
            else:
                res_img[i][j] = 0

    return res_img

class DirectoryFlow(Sequence):

    def __init__(self, x_dir_path, y_dir_path, batch_size, img_size,img_ext='.jpg', label_suffix_ext='_lab.png', skip_encoding=False, is_rgb=False, colors=None):

        self.x_dir_path = x_dir_path
        self.y_dir_path = y_dir_path

        x_list = os.listdir(x_dir_path)
        y_list = os.listdir(y_dir_path)

        self.data_names_list = [x_el.split('.')[0] for x_el in x_list if x_el.split('.')[0]+label_suffix_ext]
        self.temp_data_names = self.data_names_list.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_ext = img_ext
        self.label_suffix_ext = label_suffix_ext
        self.skip_encoding = skip_encoding
        self.is_rgb = is_rgb
        self.colors = colors

    def _get_image_batch(self, allow_duplicates=False,img_ids=None):

        colors = [(0,0,0), (151,0,255), (30,230,255), (184,115,117), (216,255,0), (252,199,0),\
         (255,0,0), (255,0,246), (140,140,140), (0,255,0), (244,255,0), (152,166,0)]

        images = []
        labels = []
        if img_ids is not None:
            image_names_list = img_ids
        else:
            if allow_duplicates:
                image_names_list = np.random.choice(self.temp_data_names, (self.batch_size))
            else:
                image_names_list = random.sample(self.temp_data_names, (self.batch_size))

        self.temp_data_names = [x for x in self.temp_data_names if x not in image_names_list]

        for i_name in image_names_list:
            image = cv2.imread(self.x_dir_path + '/' + i_name + self.img_ext)
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image*(1./255)

            label = cv2.imread(self.y_dir_path + '/' + i_name + self.label_suffix_ext)
            label = cv2.resize(label, self.img_size, interpolation=cv2.INTER_NEAREST)
            if not self.skip_encoding:
                if self.is_rgb:
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                    label = rgb_to_categorical(label, colors)
                label = categorical_encoding_to_prob(label, 12)
            else:
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

            images.append(image)
            labels.append(label)

        return (np.array(images), np.array(labels))

    def __len__(self):
        return int(math.floor(len(self.data_names_list)/self.batch_size))


    def __getitem__(self, index):

        return self._get_image_batch()

    def reset_temp_data_names(self):

        self.temp_data_names = self.data_names_list.copy()

    def on_epoch_end(self):
      self.reset_temp_data_names()

class AugDirectoryFlow(DirectoryFlow):

    def __init__(self, x_dir_path, y_dir_path, batch_size, img_size, img_ext='.jpg', label_suffix_ext='_lab.png',\
                 h_flip=False, v_flip=False, rotation_deg=0, zoom_in_scale=1.0, skip_encoding=False, is_rgb=False, colors=None):

        super().__init__(x_dir_path, y_dir_path, batch_size, img_size, img_ext, label_suffix_ext, skip_encoding, is_rgb, colors)
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotation_deg = rotation_deg
        if(zoom_in_scale<1.0):
            raise ValueError('Zoom_in_scale is expected to be 1.0 or higher')
            zoom_in_scale = 1.0
        self.zoom_in_scale = zoom_in_scale

    def _apply_augmentations(self,data_pairs):
        images, labels = data_pairs
        for i in range(len(images)):
            if(self.v_flip):
                if(np.random.randint(2)>0):
                    images[i] = cv2.flip(images[i],0)
                    labels[i] = cv2.flip(labels[i],0)
            if(self.h_flip):
                if(np.random.randint(2)>0):
                    images[i] = cv2.flip(images[i],1)
                    labels[i] = cv2.flip(labels[i],1)
            if(self.rotation_deg > 0 or self.zoom_in_scale > 1.0):
                degrees = 0
                if(self.rotation_deg > degrees):
                    degrees = np.random.randint(0,self.rotation_deg+1)

                scale = 1.0
                if(self.zoom_in_scale > scale):
                    scale = np.random.uniform(1.0,self.zoom_in_scale)

                height, width, _ = images[i].shape
                rotation = cv2.getRotationMatrix2D((height/2, width/2),\
                                                   degrees,scale)
                images[i] = cv2.warpAffine(images[i], rotation, (width, height))
                labels[i] = cv2.warpAffine(labels[i], rotation, (width, height))
        return (images, labels)


    def __getitem__(self, index):

        data_pairs = super()._get_image_batch()
        data_pairs = self._apply_augmentations(data_pairs)
        self.reset_temp_data_names()
        return data_pairs

    def reset_temp_data_names(self):

        super().reset_temp_data_names()

    def on_epoch_end(self):
        pass

class SkipConnLayer(Layer):

    def __init__(self, storage, key):

        super(SkipConnLayer, self).__init__()
        self.storage = storage
        self.key = key

    def call(self, inputs):
        self.storage[0][self.key] = tf.identity(inputs)
        return inputs
    
class AttentionLayer(Layer):

    def __init__(self, storage, key, num_of_filters):

        super(AttentionLayer, self).__init__()
        self.storage = storage
        self.key = key
        self.num_of_filters = num_of_filters
        self.prev_layer_conv = Conv2D(num_of_filters, (1,1), activation='relu',\
                                      kernel_initializer='he_normal')
        self.skip_conv = Conv2D(num_of_filters, (1,1), strides=(2,2),\
                                activation='relu', kernel_initializer='he_normal')
        self.post_add_activation = Activation(activation='relu')
        # TODO: check if number of filters here should really be 1 (my guess it should be the same as skip_input.shape[3])
        self.proj_conv = Conv2D(1, (1,1), activation='relu',\
                                kernel_initializer='he_normal')
        self.pre_upsample_activation = Activation(activation='sigmoid')

    def call(self, inputs):
        skip_input = self.storage[0][self.key]
        inputs_temp = self.prev_layer_conv(inputs)
        skip_temp = self.skip_conv(skip_input)
        temp_features = tf.add(inputs_temp, skip_temp)
        temp_features = self.post_add_activation(temp_features)
        temp_features = self.proj_conv(temp_features)
        temp_features = self.post_add_activation(temp_features)
        temp_features = tfimg.resize(temp_features, (skip_input.shape[1],\
                                                     skip_input.shape[2]),\
                                     method='bilinear')
        return tf.math.multiply(skip_input, temp_features)
    
class UNetSegmentModelOld:

    skip_conn_data = np.array([{}])
    dataset_classes_names = {'Background':0, 'Debris':1, 'Water':2,'Building_No_Damage':3, 'Building_Minor_Damage':4,\
                         'Building_Major_Damage':5,'Building_Total_Destruction':6, 'Vehicle':7, 'Road':8,'Tree':9, 'Pool':10, 'Sand':11}

    def create_model(self, filter_number_list, initial_shape):
        input = Input(initial_shape)
        x = input
        for num_of_filters in filter_number_list[:-1]:
            x = Conv2D(num_of_filters, (3,3), padding='same',\
                       activation='relu', kernel_initializer='he_normal')(x)
            x = Dropout(0.05)(x)

            x = Conv2D(num_of_filters, (3,3), padding='same',\
                             activation='relu', kernel_initializer='he_normal')(x)
            x = SkipConnLayer(self.skip_conn_data, f'data_{num_of_filters}')(x)
            x = MaxPooling2D((2,2))(x)

        # The lowermost layers, after this we'll do the deconv and concatenation.
        x = Conv2D(filter_number_list[-1], (3,3), padding='same',\
                   activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.05)(x)
        x = Conv2D(filter_number_list[-1], (3,3), padding='same',\
                   activation='relu', kernel_initializer='he_normal')(x)

        for num_of_filters in reversed(filter_number_list[:-1]):

            attention_x = AttentionLayer(self.skip_conn_data,\
                                            f'data_{num_of_filters}', x.shape[3])(x) #skip_conn_data[0][f'data_{num_of_filters}']
            x = Conv2DTranspose(num_of_filters, (2,2), strides=(2,2),\
                                padding='same')(x)
            x = tf.keras.layers.concatenate([attention_x, x], -1)
            x = Conv2D(num_of_filters, (3,3), padding='same',\
                       activation='relu', kernel_initializer='he_normal')(x)
            x = Dropout(0.05)(x)
            x = Conv2D(num_of_filters, (3,3), padding='same',\
                       activation='relu', kernel_initializer='he_normal')(x)

        output = Conv2D(len(self.dataset_classes_names), (1,1), padding='same',\
                        activation='softmax')(x)
        self.model = Model(inputs=input, outputs=output)

@st.cache_resource
def unetweightsreload(model_weights):
    link = 'https://drive.google.com/uc?id=1vUCcpUNfjyvRLupWX5JJiWA68ru3_Dxo'
    download_model(link,'model_weights.weights.h5')
    img_size = (480, 360)
    unetObj = UNetSegmentModelOld()
    unetObj.create_model([64, 128, 256, 512], (img_size[1], img_size[0], 3))
    unetObj.model.load_weights(model_weights)
    return unetObj.model


def load_images(folder):
    images = []
    filenames = []
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            images.append(img_path)
            filenames.append(file)
    return images, filenames

def show_image(image_path,img_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,img_size)  # Resize as needed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def preprocess_image(selected_files):
    test_ids = [i.split('.')[0] for i in selected_files]
    test_gen = RescuenetDataset(test_ids, 'images/test-org-img', 'images/test-label-img')
    return test_gen

st.title("Disaster-Resilient Military Base Damage Assessment with Autonomous Object Tracking")

# Model selection
model_choice = st.radio(
    "Choose the model to use:",
    ('UNet', 'Attention_UNet','PSPNet'),
    index=0,
    horizontal=True
)

st.markdown(f"### 📌 Selected Model: `{model_choice}`")

if model_choice == 'UNet':
    model = load_model(r'unet_rescuenet.h5')
elif model_choice == 'PSPNet':
    # Set framework for segmentation_models
    sm.set_framework('tf.keras')
    sm.framework()
    link='https://drive.google.com/uc?id=1h6F7Poose1uijNZLJJH0rHErD43sGyb0'
    download_model(link,'pspnet_rescuenet.h5')
    model = load_model('pspnet_rescuenet.h5',custom_objects={'iou_score': sm.metrics.iou_score})
elif model_choice == 'Attention_UNet':
    model_weights = 'model_weights.weights.h5'

images, filenames = load_images(IMAGE_FOLDER)

selected_files = st.multiselect("Select images to run prediction on:", filenames)

if selected_files:
    st.subheader("Selected Images")

    images_to_predict = []
    cols = st.columns(len(selected_files))

    for idx, file in enumerate(selected_files):
        path = os.path.join(IMAGE_FOLDER, file)
        if model_choice=='Attention_UNet':
            img_size=(480,360)
            test_flow = DirectoryFlow(IMAGE_FOLDER, 'images/test-label-img', 1, img_size)
            (img, label) = test_flow._get_image_batch(img_ids=[file.split('.')[0]])
        elif model_choice=='PSPNet':
            image_dir=IMAGE_FOLDER
            mask_dir="images/test-label-img"
            image_ids=[file.split('.')[0]]
            img = RescuenetDataset(image_dir,mask_dir,image_ids,img_size=(384,384))
        else:
            img = show_image(path,(256, 256))
        images_to_predict.append(img)
        cols[idx].image(path, caption=file, use_container_width=True)

    if st.button("Run Prediction"):

        start_time = time.time()
        if model_choice=='Attention_UNet':
            model = unetweightsreload(model_weights)
            predictions = [model.predict(img) for img in images_to_predict]
        elif model_choice=='PSPNet':
            predictions=[]
            for img in images_to_predict:
                sample_img, sample_mask = img[0]
                predictions.append(model.predict(np.expand_dims(sample_img[0], axis=0)))
        else:
            predictions = model.predict(np.array(images_to_predict))
        elapsed_time = time.time() - start_time

        st.subheader("🔍 Prediction Results")

        # To accumulate metrics across samples
        all_gt = []
        all_pred = []
        # Class labels (0 to 11)
        NUM_CLASSES = 12
        CLASS_NAMES = ['Background', 'Debris', 'Water', 'Building_No_Damage', 'Building_Minor_Damage',
                    'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle', 'Road', 'Tree', 'Pool', 'Sand']
        for fname, pred in zip(selected_files, predictions):
            testname = f"{fname.split('.')[0]}_lab.png"
            mask_path = os.path.join("images/test-label-img", testname)
            img_path = os.path.join(IMAGE_FOLDER, fname)

            # Load original image
            orig_image = Image.open(img_path)

            # Load ground truth mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            class_colors = [
                    (0, 0, 0),             # 0 - background
                    (7,3,252),             #debris
                    (61, 230, 250),        # 1 - water
                    (180, 120, 120),       # 2 - building-no-damage
                    (235, 255, 7),         # 3 - building-minor-damage
                    (255, 184, 6),         # 4 - building-major-damage
                    (255, 0, 0),           # 5 - building-total-destruction
                    (255, 0, 245),         # 6 - vehicle
                    (140, 140, 140),       # 7 - road
                    (4, 250, 7),           # 8 - tree
                    (255, 235, 0),          # 9 - pool
                    (160, 150, 20)        # 10 - sand
                ]

            # Normalize RGB values from 0–255 to 0–1 for matplotlib
            normalized_colors = np.array(class_colors) / 255.0
            cmap = ListedColormap(normalized_colors)
            norm = BoundaryNorm(np.arange(len(class_colors) + 1), cmap.N)

            # Now apply to your masks:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(orig_image)
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(mask, cmap=cmap, norm=norm)
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis('off')
            
            if model_choice=='Attention_UNet' or model_choice=='PSPNet':
                # Process predicted mask
                if isinstance(pred, np.ndarray):
                    pred = pred.squeeze()
                    pred = np.argmax(pred, axis=-1) 

                axs[2].imshow(pred, cmap=cmap, norm=norm)
                axs[2].set_title("Predicted Mask")
                axs[2].axis('off')

            else:
                # Process predicted mask
                if isinstance(pred, np.ndarray):
                    pred = pred.squeeze()
                    if pred.max() <= 1.0:
                        pred = (pred * 255).astype(np.uint8)


                axs[2].imshow(pred, cmap=cmap, norm=norm)
                axs[2].set_title("Predicted Mask")
                axs[2].axis('off')

            plt.tight_layout()
            
            # Convert matplotlib figure to image buffer for Streamlit
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf, caption=f"Results for {fname}", use_container_width=True)
            plt.close()
            
            #gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize prediction to GT size if needed
            if pred.shape != mask.shape:
                pred = cv2.resize(pred.squeeze(), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            all_gt.append(mask.flatten())
            all_pred.append(pred.flatten())

        # Flatten all masks
        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        # Calculate per-class and macro-averaged scores
        iou_scores = jaccard_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        precision_scores = precision_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        recall_scores = recall_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        f1_scores = f1_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)

        # Mean scores
        mean_iou = np.mean(iou_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)

        st.subheader("📈 Segmentation Metrics Per Class")

        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "IoU": iou_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1-Score": f1_scores
        }).round(4)

        st.dataframe(df)

        st.markdown("### 🔍 Macro-Averaged Scores")
        st.write(f"**Mean IoU:** {mean_iou:.4f}")
        st.write(f"**Mean Precision:** {mean_precision:.4f}")
        st.write(f"**Mean Recall:** {mean_recall:.4f}")
        st.write(f"**Mean F1-Score:** {mean_f1:.4f}")


        st.markdown("---")
        st.metric(label="🕒 Prediction Time", value=f"{elapsed_time:.2f} seconds")
        st.metric(label="📊 Number of Images", value=len(selected_files))

else:
    st.info("Please select at least one image to run prediction.")
