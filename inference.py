from keras.preprocessing.image import img_to_array
from keras.applications.densenet import DenseNet169, DenseNet121, preprocess_input,decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
import pandas as pd
from tqdm import tqdm
import os
import scipy as sp
import keras
import numpy as np
import tensorflow as tf
import random
from keras.optimizers import Adam
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
import requests
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image/=255.0
    image = np.expand_dims(image, axis=0)
    return image


def prediction(image1,model):
    yhat = model.predict(image1)
    if yhat>0.5:
        label='Abnormal'
    else:
        label='Normal'
    return label,yhat

def predict_bodypart(image1,model):
    yhat = model.predict(image1)
    class_idx=np.argmax(yhat)
    class_map=['ELBOW','FINGER','FOREARM','HAND','HUMERUS','SHOULDER','WRIST']

    return class_map[class_idx],max(yhat)
