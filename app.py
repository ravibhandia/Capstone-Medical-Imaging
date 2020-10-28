
from keras.applications.densenet import DenseNet169, DenseNet121, preprocess_input,decode_predictions
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense

from flask import Flask, render_template, request
import flask
from inference import prediction,prepare_image,predict_bodypart,return_grad_CAM_heatmap
from PIL import Image
import io
import os
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def build_model(n_classes,model_type,body_part):
    model = Sequential()
    model.add(DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='sigmoid'))
    model.load_weights('weights/best_' + model_type + '_' + body_part.lower() + '_classifier_weights.best.hdf5')
    return model


@app.route('/',methods=['GET','POST'])
def predict():
    data={'success':False}

    if flask.request.method=='GET':
        return render_template('index.html',value='hi')

    elif flask.request.method=='POST':
        if flask.request.files.get('file'):

            image=flask.request.files['file'].read()
            image=Image.open(io.BytesIO(image))

            image=prepare_image(image)

            #1. Predict Body Part

            body_part,yhat_body=predict_bodypart(image,body_part_model)

            #2. Predict Abnormality
            label,yhat=prediction(image,model_dict[body_part])

            #3 Get GradCAM Heatmap
            # For densenet 592 is the last conv_layer, 595 and 596 is the last pooling layer and the last dense layer respectively.

            #heatmap=make_gradcam_heatmap(image,model_dict[body_part],model_dict[body_part].get_layer(index=0).get_layer(index=592),[model_dict[body_part].get_layer(index=0).get_layer(index=595),model_dict[body_part].get_layer(index=2)])

            data['predictions']=[]
            data['predictions'].append({'body_part':body_part,'yhat_body':yhat_body,'label':label,'yhat':yhat})
            data['success']=True
    return render_template('result.html', body_part=data['predictions'][0]['body_part'], label=data['predictions'][0]['label'],yhat=data['predictions'][0]['yhat'])


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to suppress GPU messages
    list_of_bodyparts=['ELBOW','FINGER','FOREARM','HAND','HUMERUS','SHOULDER','WRIST']
    global model_dict
    model_dict={}

    for part in list_of_bodyparts:
        model_dict[part]=build_model(1,'densenet',part)
    global body_part_model
    body_part_model =build_model(7,'densenet','bodypart')

    app.run(host='0.0.0.0', port=80)