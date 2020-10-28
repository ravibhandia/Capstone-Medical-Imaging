# Muscoskeletal X-Ray Abnormality Classifier with Grad-CAM

The purpose of this project is to create a webapp that can classify X-Ray images as abnormal with a high rate of specificity and sensitivity. And to provide a level of interpretability that is currently possible with convolutional neural networks, using gradient classification activation maps (Grad- CAM) by superimposing a heatmap ontop of the original image.

The data used in this project  and the idea of building abnormality detectors for muscoskeletal radiographs came from [Rajpurkar et al.](https://arxiv.org/abs/1712.06957)
You can retrieve the dataset by going to the competition leaderboard and requesting it [here.](https://stanfordmlgroup.github.io/competitions/mura/)

All training of the 7 body part models and the body detector model can be found in the folder [/Notebooks](https://github.com/ravibhandia/Capstone-Medical-Imaging/tree/master/Notebooks). Additionally an attempt was also made to try and build a generalized body_part anomaly detector model that would be fed all the data with only abnormality labels to see how it would perform.

## 1. Loading and preprocessing of data

The image data came in .jpg files. Metadata .csv files were used to create Pandas dataframes that included the image paths, patient no, study no, and the label of abnormality. The dataset contains ~36000 images in the training set and 3197 images in the validation set.

An initial Data Exploration including the count of the number of images by body_part can be found in [Data Exploration.ipynb](https://github.com/ravibhandia/Capstone-Medical-Imaging/blob/master/Notebooks/Data%20Exploration.ipynb)

Using the information in the metadataframes provided, new dataframes were formed that contained the labels, image paths, and body part information. A separate notebook

One bug I found while dealing with preprocessing in deployment is that load_img in keras.preprocessing version 1.12 has significant problems dealing with io.Bytes.io object type, so I downgraded to 1.10 and it works perfectly.




## 2. Data Modeling and Training

Lots of data modeling experimentation was done as can be see in each of the notebooks to see how to best build initially a model that had the capability to overfit and later on a model that could generalize over the distribution. Eventually, Densenet (as suggested by Rajpurkar's paper) and Efficientnet B0 were used.

Grad-cam was only implemented for DenseNet and in the streamlit implementation files which can be seen in [main.py](https://github.com/ravibhandia/Capstone-Medical-Imaging/blob/master/main.py) and [classify.py](https://github.com/ravibhandia/Capstone-Medical-Imaging/blob/master/classify.py).

All the training of the models was done on Google Colab Pro for GPU access and storage of the data.

## 3. Performance

Example Metrics for DenseNet 169 Layer Performance on Validation Set of the Wrist

Confusion Matrix

| 336 | 28  |
|-----|-----|
| 99  | 196 |



Classification Report
              
              
              precision    recall  f1-score   support

      Normal       0.77      0.92      0.84       364
    Abnormal       0.88      0.66      0.76       295

    accuracy                           0.81       659
    macro avg       0.82      0.79      0.80       659
    weighted avg       0.82      0.81      0.80       659

AUC and ROC Curve:
![ROC Curve](/static/AUC_ROC_Curve.png "ROC Curve")

## 4. Deployment

Deployment was done in two ways: Streamlit and AWS ECS

First deployment on streamlit was implemented by main.py and classify.py. As you can see below this implementation has Grad-CAM implemented on it.

![Streamlit Example](/static/Streamlit_example.png "Streamlit Example")

To run the streamlit app locally you just need to use the command:

 >streamlit run main.py

The next form of deployment was using a flask to serve html on AWS. The python scripts used for this are app.py and inference.py

The first step was to build a requirements.txt which includes all the packages used in this project. Next, the implementation was tested locally by using the command 
   >python app.py

in shell

You will see this 

![Flask home](/static/Flask_main.png "Flask Home")

When you upload an x-ray image it will send you to a prediction page, showing you this below:

![Flask prediction](/static/Flask_prediction.png "Flask Prediction")

Next a docker file was written to containerize the app. To build the docker image the following command was used:

 >docker build -t username/x_ray_classifier .

Remember the period!!

Now you can run a container of the image using the following command:

 >docker run -p 80:80 username/x_ray_classifier

You can also confirm which docker images and containers are running by using the following commands

   >docker image ls

   >docker ps -a

next use docker login to authenticate yourself:

   >docker login register-123.docker.io

and then push your docker container

  >docker push username/x_ray_classifier

Once my docker image was on docker hub. I then used the URL that points to my image on docker hub: docker.io/username/x-ray and then used it to host on AWS Elastic Container Service. You can repeat the instruction using this [article](https://reflectoring.io/aws-deploy-docker-image-via-web-console/).

This X-Ray abnormality detector can then be acessed from the browser with the url specified by AWS ECS.

## Acknowledgements and Resources Used

1. AWS ECS Deployment Guide from Reflectoring.io https://reflectoring.io/aws-deploy-docker-image-via-web-console/
2. Grad-CAM example with keras https://keras.io/examples/vision/grad_cam/
3. Streamlit upload feature https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0
4. MURA Dataset at Stanford Machine Learning Group https://stanfordmlgroup.github.io/competitions/mura/
