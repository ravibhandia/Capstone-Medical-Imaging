#python3 -m pip install whatever
import streamlit as st
from PIL import Image
from classify import predict_heatmap,preprocess_image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from keras.preprocessing.image import array_to_img,img_to_array


st.title('X-Ray Classifier')
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,heatmap,image = predict_heatmap(uploaded_file,1)

    st.write(float(label[0][0]*100))
    #plt.imshow(image.reshape((224,224,3)), alpha=0.7);
    #plt.imshow(heatmap, cmap='jet', alpha=1)
    image=image.reshape((224,224,3))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)


    #scale 0.01 however you need for the particular image to be readable, I chose 0.01
    superimposed_img = jet_heatmap * 0.01 + image
    superimposed_img = array_to_img(superimposed_img).resize((700,700))

    st.image(superimposed_img)

    plt.axis('off')
    st.pyplot()


    #st.write('%s (%.2f%%)' % (label[1], label[2] * 100))