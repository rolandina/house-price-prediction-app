import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit.state.session_state import Value
from model.data import Data 

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

### models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
### model selection
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split, cross_val_score
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#prepare cache data
@st.cache
def load_data():
    return Data()

target = "SalePrice"

def add_parameter_ui(reg_name):
    params = dict()
    if reg_name == 'Linear Regression':
        normalize = st.sidebar.selectbox('normalize',('True', 'False')); params['normalize'] = normalize

    elif reg_name == 'Lasso':
        alpha = st.sidebar.slider('alpha', min_value = 1, max_value = 30); params['alpha'] = alpha
        max_iter = st.sidebar.slider('maximum iteration', min_value = 500, max_value = 3000, step = 500); params['max_iter'] = max_iter
        tol = st.sidebar.slider('tolerance', min_value = 1.e-4, max_value = 1.e-1, value = 1.e-2, format = '%e' ); params['tol'] = tol

    elif reg_name == 'Ridge':
        alpha = st.sidebar.slider('alpha', min_value = 1, max_value = 30); params['alpha'] = alpha
        max_iter = st.sidebar.slider('maximum iteration', min_value = 500, max_value = 3000, step = 500); params['max_iter'] = max_iter
        solver = st.sidebar.selectbox('solver',('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg')); params['solver'] = solver
        tol = st.sidebar.slider('tolerance', min_value = 1.e-4, max_value = 1.e-1, value = 1.e-2, format = '%e' ); params['tol'] = tol
        
    return params

def add_parameter_gs_ui(reg_name):
    params = dict()
    if reg_name == 'Linear Regression':
        normalize = st.sidebar.multiselect('normalize',['True', 'False'], default=['True']); params['normalize'] = normalize

    elif reg_name == 'Lasso':
        alpha = st.sidebar.slider('alpha', min_value = 1, max_value = 30, value=(1, 5)); params['alpha'] = list(np.linspace(alpha[0], alpha[1], num = alpha[1]-alpha[0]+1, endpoint=True))
        max_iter = st.sidebar.slider('maximum iteration', min_value = 500, max_value = 3000, step = 500, value = (500,2000)); params['max_iter'] = max_iter
        tol = st.sidebar.multiselect('tolerance',[1.e-4, 1.e-3, 1.e-2, 1.e-1],  default = [1.e-2, 1.e-1] ); params['tol'] = tol
        st.write(params['tol'])
        st.write(type(params['tol']))
    elif reg_name == 'Ridge':
        alpha = st.sidebar.slider('alpha', min_value = 1, max_value = 30, value=(1, 10)); params['alpha'] = alpha
        max_iter = st.sidebar.slider('maximum iteration', min_value = 500, max_value = 3000, step = 500, value=(500, 2000)); params['max_iter'] = max_iter
        solver = st.sidebar.multiselect('solver',['svd', 'cholesky', 'lsqr', 'sparse_cg'], ['svd', 'lsqr']); params['solver'] = solver
        tol = st.sidebar.slider('tolerance', min_value = 1.e-4, max_value = 1.e-1,  value = (1.e-4, 1.e-1), format = '%e' ); params['tol'] = tol
        st.write(params['tol'])
        st.write(type(params['tol']))
    return params

def get_regressor(reg_name, params):
        reg = None
        if reg_name == 'Linear Regression':
            reg = LinearRegression(normalize=params['normalize'])
        elif reg_name == 'Lasso':
            reg = Lasso(alpha=params['alpha'], max_iter=params['max_iter'], tol=params['tol'])
        else:
            reg = Ridge(alpha=params['alpha'], max_iter=params['max_iter'], solver= params['solver'], tol=params['tol'] )
        return reg


def show_test_metrics(model, X_test, y_test):
    # Get predictions
    predictions = model.predict(X_test)

    # Show metrics
    st.write('Metrics of y_test prediction:')
    mse = mean_squared_error(y_test, predictions)
    st.write("MSE:", mse)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse)
    r2 = r2_score(y_test, predictions)
    st.write("R2:", r2)

    # Plot predicted vs actual
    
    fig = plt.figure()
    sns.scatterplot(x= y_test/1000, y= predictions/1000)
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title('Predicted vs Actual prices,  in thousands $ ')
    z = np.polyfit(y_test/1000, predictions/1000, 1)
    p = np.poly1d(z)
    sns.lineplot(x=y_test/1000,y= p(y_test/1000), color='red')
    #plt.grid( color='r', linestyle='-', linewidth=2)
    st.pyplot(fig)

def app():
    """ This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """
    #prepare data
    data = load_data()
    test = data.get_prepared_test_data()
    train = data.get_prepared_train_data()

    X, y = train[[col for col in train.columns if col != target]], train[target]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20, random_state=1)

    #separate page on 3 columns - left column - for 'building a model section', middle one for spacing and right one - for 'price prediction'
    col1, col_space, col2 = st.beta_columns([4,1,4])

    with col1:
        st.title("Building a model")
        model_name = st.selectbox('Select models:',
            ('Linear Regression', 'Ridge', 'Lasso'))


        is_grid_search = st.checkbox('Grid Search')

        if is_grid_search:
            params = add_parameter_gs_ui(model_name)
            st.write(model_name, " with grid search")
            st.write(params)



        else:
            params = add_parameter_ui(model_name)
            model = get_regressor(model_name, params)
            model.fit(X_train, y_train)
            show_test_metrics(model, X_test, y_test)

    # right column with house price prediction
    with col2:
        col2.title("Price Prediction")

        





    





    