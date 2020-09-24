#python3 -m pip install whatever
import streamlit as st
from PIL import Image
from classify import predict_heatmap
import matplotlib.pyplot as plt


st.title('X-Ray Classifier')
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,heatmap,image1 = predict_heatmap(uploaded_file,1)
    st.write(float(label[0][0]*100))
    plt.imshow(image1, alpha=0.7);
    plt.imshow(heatmap, cmap='jet', alpha=0.3)
    plt.axis('off')
    st.pyplot()


    #st.write('%s (%.2f%%)' % (label[1], label[2] * 100))