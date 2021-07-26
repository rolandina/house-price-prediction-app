import os
import streamlit as st
import numpy as np
from PIL import  Image
 

# Custom imports 
from multipage import MultiPage

from pages import test, data_analysis, model_and_prediction, pca # import your pages here

display = Image.open('data/house_price_b.jpg')

st.set_page_config(page_title="House Price Prediction App", page_icon=display, layout='wide', initial_sidebar_state='auto')
# Create an instance of the app 
app = MultiPage()

# Title of the main page

display = np.array(display)
#st.image(display, width = 200)

col1, col2 = st.beta_columns([1,4])
col1.image(display)#, width = 200)
col2.title("House Price Prediction Application")

col2.markdown("### With our house price prediction \
application you are free to analyse the price of the houses depending from different paremeters as long as to predict the price of the house by choosing parameters you need.")



# Add all your application here
#app.add_page("test", test.app)
app.add_page("Data Analysis", data_analysis.app)
app.add_page("PCA", pca.app)
app.add_page("Model and price prediction", model_and_prediction.app)


# The main app
app.run()

