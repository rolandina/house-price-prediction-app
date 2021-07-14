import streamlit as st
import pandas as pd


def app():
    """This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """

    st.title("PCA Theory")
    st.header("1")
    st.header("2")

    st.code('''y=4''', language="python")

    st.dataframe()


    st.graphviz_chart('''
        digraph {
            data -> api
            api -> model
            model -> prediction
            api -> prediction
            api -> streamlit
        }
    ''')



    st.graphviz_chart('''
        digraph {
            pipeline -> scaler
            scaler -> pca
            pca -> model
        }
    ''')

    #checkbox
    agree = st.checkbox('I agree')

    if agree:
        st.write('Great!')

    #radio
    genre = st.radio(
        "What's your favorite movie genre",
        ('Comedy', 'Drama', 'Documentary'))
    if genre == 'Comedy':
        st.write('You selected comedy.')
    else:
        st.write("You didn't select comedy.")

    #selectbox
    option = st.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone'))

    st.write('You selected:', option)

    #multiselect

    options = st.multiselect(
        'What are your favorite colors',
        ['Green', 'Yellow', 'Red', 'Blue'],
        ['Yellow', 'Red'])
    st.write('You selected:', options)

    #select slider
    start_color, end_color = st.select_slider(
        'Select a range of color wavelength',
        options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        value=('red', 'blue'))
    st.write('You selected wavelengths between', start_color, 'and', end_color)


    # text and number input

    name = st.text_input("Type your name")
    age = st.number_input("Type your age", min_value=1, max_value=120, step=1)
    if name is not None:
        st.write("My name is", name, "and I am", age, "years old." )


    with st.form("my_form"):
        st.write("Inside the form")
        slider_val = st.slider("Form slider")
        checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

    st.write("Outside the form")

    #st.balloons()

    st.sidebar.slider("choose k", 0, 10, 1)

