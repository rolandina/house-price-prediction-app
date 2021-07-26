columns_to_save = [
       'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'
       ]

ordinal_cols = [
            'PoolQC', 'KitchenQual', 'ExterQual', 'HeatingQC', 'BsmtCond', 'BsmtQual',
            'ExterCond', 'GarageCond', 'GarageQual', 'FireplaceQu', 'Neighborhood']

numeric_power_cols = [
            'Id', 'LotArea', 'GrLivArea', 'BsmtUnfSF', '1stFlrSF', 'TotalBsmtSF',
            'BsmtFinSF1', 'GarageArea', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF',
            'OpenPorchSF', 'BsmtFinSF2', 'EnclosedPorch', 'YearBuilt', 'LotFrontage',
            'ScreenPorch']

numeric_scale_cols = [
            'LowQualFinSF', 'MiscVal', '3SsnPorch', 'MSSubClass', 'MoSold',
            'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'PoolArea', 'BedroomAbvGr',
            'YrSold', 'GarageCars', 'KitchenAbvGr', 'Fireplaces', 'BsmtFullBath',
            'FullBath', 'BsmtHalfBath', 'HalfBath'
        ]



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

### models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


def get_df_from_heroku(url):
    """Function takes link to api to get data frame.
    Returns data frame."""
    response = requests.get(url)
    a_json = json.loads(response.json())
    return pd.DataFrame.from_dict(a_json, orient="columns")


class Data:
    # change parameters here
    target = 'SalePrice'
    __chosen_columns = columns_to_save
    __prediction_columns = []

 
        #version2 (manual)
        #columns_to_save = ['LotArea', 'OverallQual', '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MSZoning', 'LotShape', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleType', 'SaleCondition', "SalePrice"]
        #1. create list of features to drop if they are not in a save list


    def __init__(self):
        ## initialise with Data object with data frame
        #__df_train, __df__test - initial data frame without preprocessing 
        # df_train for building model and training model, X_predict for prediction only
        self.__df_train, self.__df_test = self.__read_df()
        #divide only train set on X and y as for test set we don't have target
        self.__X, self.__y = self.__df_train.drop(columns = [self.target]), self.__df_train[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.__X, self.__y, test_size=0.20, random_state=1)
        columns_to_save =  create_best_corr_list(self.__df_train, self.target) + ['Neighborhood']


    # read data file
    def __read_df(self):
        url_heroku = "https://house-price-fastapi.herokuapp.com"
        url_train = url_heroku + "/get_train_set"
        url_test = url_heroku + "/get_test_set"

        df_train = get_df_from_heroku(url_train)
        df_test = get_df_from_heroku(url_test)
        return (df_train, df_test)
  
    def get_train_df(self):
        return self.__df_train

    def get_test_df(self):
        return self.__df_test

    def get_columns_names(self):
        """Return list of features used in a model (use for prediction)"""
        return self.__prediction_columns
    
    def get_prepared_data_for_prediction(self, X_input):
         # do my transformation pipeline
        X_output = self.__preprocessing_standard(X_input)
        return X_output

    def get_prepared_data_for_model(self, model_name):
        if model_name == 'RandomForest':
            #do pca transformation
            X_train = self.__preprocessing_pca(self.X_train)
            X_test = self.__preprocessing_pca(self.X_test)          
        else:
            # do my transformation pipeline
            #version1
            corr_cols =  create_best_corr_list(self.__df_train, self.target) + ['Neighborhood']
            self.__prediction_columns = corr_cols

            X_train = self.__preprocessing_standard(self.X_train)
            X_test = self.__preprocessing_standard(self.X_test)
        return (X_train, X_test, self.y_train, self.y_test)

    def __preprocessing_pca(self, X_input):
        #apply pipeline here for pca transformation
        #for pca models
        X_input = X_input.drop(['Id' , 'MSSubClass', 'OverallQual', 'OverallCond', 'MoSold'], axis=1)
        X_input = X_input.select_dtypes(include=['int', 'float'])
        self.__prediction_columns = X_input.columns.to_list()

        numeric_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        return numeric_pipe.fit_transform(X_input)

    def __preprocessing_standard(self, X_input):

        drop_features = [col for col in X_input.columns if col not in self.__prediction_columns] 
        if self.target in drop_features:
            drop_features.remove(self.target)

        #2. ordinal - categorical
        ordinal_features = [
            feature for feature in ordinal_cols if feature in self.__prediction_columns
        ]
        ordinal_cat = create_categories_for_ordinal_features(self.__df_train, self.target, ordinal_features)

        #3 numerical - with log transformation    
        numeric_power_features = [
            feature for feature in numeric_power_cols
            if feature in self.__prediction_columns
        ]

        #4 numerical without log transformation
        numeric_scale_features = [
            feature for feature in numeric_scale_cols
            if feature in self.__prediction_columns
        ]

        #5 categorical - dummies
        categorical_features = self.__df_train.select_dtypes(include=['object']).columns
        categorical_features = [
            feature for feature in categorical_features
            if feature not in ordinal_features and feature in self.__prediction_columns
        ]

        numeric_power_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                #('powtransform', PowerTransformer()),
                ('scaler', StandardScaler())
                ])
        numeric_scale_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                ('scaler', StandardScaler()),          
                ])

        ordinal_transformer = Pipeline(steps=[
            ('imputer1', SimpleImputer(strategy='constant', fill_value='absent')), 
            ('imputer2', SimpleImputer(missing_values = None, strategy='constant', fill_value='absent')),
            ('ordenc', OrdinalEncoder(categories = ordinal_cat)),
            ('scaler', MinMaxScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer1', SimpleImputer(                       strategy='constant', fill_value='absent')), 
            ('imputer2', SimpleImputer(missing_values = None, strategy='constant', fill_value='absent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[
            ('drop', 'drop', drop_features),
            ('num_pow', numeric_power_transformer, numeric_power_features),
            ('num_scal', numeric_scale_transformer, numeric_scale_features),
            ('cat', categorical_transformer, categorical_features),
            ('ordinal', ordinal_transformer, ordinal_features)
        ])

        preprocessor.fit(self.X_train[X_input.columns])

        return preprocessor.transform(X_input)

#helf function for standard preprocessing pipeline 
def create_categories_for_ordinal_features(df, target, list_ordinal_features):
    ordinal_categories = []
    for feature in list_ordinal_features:
        categ = df[[feature, target
                    ]].groupby(by=feature, as_index=False).mean().sort_values(
                        by=target, ascending=True)[feature].to_list()
        ordinal_categories.append(categ)
    return ordinal_categories


def create_best_corr_list(df, target, min_corr_coef = 0.51):
    """
    create_best_corr_list(df, target, min_corr_coef) - function which returns a list with numerical features 
    which correlation coef with target more than 0.51. 
    Also removes tweans features - features which are strongly correlated one with another.
    
    Parameters:
    df - data frame
    target - target column
    min_corr_coef - min correlation allowed between target and features, by default - 0.51
    Return:
    <list>  - list of features 
    """
    
    corrmat = df.corr()
    best_corr_cols = corrmat[corrmat[target]>min_corr_coef].sort_values(by=target, ascending = False)
    best_corr_cols = best_corr_cols.index.to_list()
    best_corr_cols.remove(target)
    
    for col1 in best_corr_cols:
        for col2 in best_corr_cols:
            if col1 != col2:
                corr12 = corrmat.loc[col1, col2]
                if corr12 > 0.80: ## assumtion for tweans
                    if corrmat.loc[col1, target] >= corrmat.loc[col2, target]:
                        best_corr_cols.remove(col2)
                    else:
                        best_corr_cols.remove(col1)
                        
    return best_corr_cols    
