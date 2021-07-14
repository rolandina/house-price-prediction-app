import streamlit as st
import pandas as pd
from streamlit.state.session_state import Value


def app():
    """This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """

    st.title("Building a model")
    model_type = st.selectbox('Select models:',
        ('Linear Regression', 'Ridge', 'Lasso'))

    st.write("current model is:", model_type)

    is_grid_search = st.checkbox('Grid Search')

    if is_grid_search:
        st.write('Show grid search parameters')
        K = st.slider('Select a range for parameter k',  0.0, 100.0, (25.0, 75.0))
        st.write(K)
    else:
        st.write("show parameters")


    