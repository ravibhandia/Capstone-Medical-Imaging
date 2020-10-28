from keras.preprocessing.image import img_to_array
import numpy as np
import ssl
import keras
import tensorflow as tf
from eli5.keras import explain_prediction

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

def return_grad_CAM_heatmap(model,img_array):
    explain_prediction(model,img_array)
