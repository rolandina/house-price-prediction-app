#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import des modules



import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn import set_config
set_config(display='diagram')


# # Import de la data

train_path = "/Users/marie/Ecole_IA/Brief_5_PCA/data/train.csv"
test_path = "/Users/marie/Ecole_IA/Brief_5_PCA/data/test.csv"

# Import des csv via pandas
X_full = pd.read_csv(train_path, index_col='Id')
X_test_full = pd.read_csv(test_path, index_col='Id')


# # EDA


# Observation de la donnée, nulls, doubles, outliers...
def df_info(df):
        len_df = len(df)
        all_columns = len(df.columns)
        all_nan = df.isna().sum().sum()
        list_of_numerics = df.select_dtypes(include = ['float','int']).columns
        print(f"""
        Longueur du dataset : {len_df} enregistrements
        Nombre de colonnes : {all_columns}
        """)
        echantillonColonnes = []
        for i in df.columns:
            listcolumn = str(list(df[i].head(5)))
            echantillonColonnes.append(listcolumn)
        obs = pd.DataFrame({'type': list(df.dtypes),
        'Echantillon': echantillonColonnes,
        "% de valeurs nulles":
        round(df.isna().sum() / len_df * 100, 2),
        'Nbr L dupliquées' : (df.duplicated()).sum(),
        'Nbr V unique' : df.nunique(),
        'Nbr Outliers' : df.apply(lambda x: sum(
                                 (x<(x.quantile(0.25) - 1.5 * (x.quantile(0.75)-x.quantile(0.25)))) |
                                 (x>(x.quantile(0.75) + 1.5 * (x.quantile(0.75)-x.quantile(0.25))))
                                 if x.name in list_of_numerics else ''))
        })
        return obs

pd.set_option('max_rows', None)
df_info(X_full)


# In[7]:


# On a 3 colonnes qui sont de types int alors que ce sont des catégories, pour plus de facilité, on transforme toutes les colonnes catégorielles en dtype='category'
for col in X_full.select_dtypes(include = ['object']):
    X_full[col] = X_full[col].astype('category')
for col in ['MSSubClass','OverallQual','OverallCond']:
    X_full[col] = X_full[col].astype('category')




# Vérification des types
X_full.dtypes



# Observation des chiffres clés des colonnes numériques (moyenne, mediane, quartiles, min, max)
X_full.describe()


# Observation des chiffres clés des colonnes catégorielles (frequence, unique...)
X_full.describe(include='category')


print(f'Total null: {X_full.isna().sum().sum()}')
print(f'Total dupliqué: {X_full.duplicated().sum()}')


# Visualisation des colonnes numériques
def hist_box(df, numerical_features_list):
    sns.set()
    sns.set_palette("Paired")
    fig, axes = plt.subplots(nrows=len(numerical_features_list), ncols=2, figsize = (20, 30), constrained_layout=True)
    for i, feature in enumerate(numerical_features_list):
        sns.histplot(data = df, x=feature, kde = True,   ax=axes[i,0])
        sns.boxplot(data = df, x=feature, ax=axes[i,1])
    plt.show()

numerics_columns = X_full.select_dtypes(include = ['float','int']).columns
hist_box(X_full, numerics_columns)



# Visualisation des corrélations et colinéarité
sns.clustermap(X_full.corr(), figsize=(30,30), annot=True)


# # Baseline : premier modèle sans pousser l'exploration, sans sélection de features...

# Définition de la target et des features
y = X_full.SalePrice
X = X_full.drop(['SalePrice'], axis=1)




# Séparation en jeu de train et de valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)



# Test du pipeline de base
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))],remainder='passthrough')

reg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', RandomForestRegressor())])


# Entrainement du modèle sur le jeu de train
reg.fit(X_train, y_train)


# Comparaison des r2 train et valid
print(reg.score(X_train, y_train), reg.score(X_valid, y_valid))


# Prediction et metrics avec le valid
y_pred = reg.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f'MSE: {mse}')
mae = mean_absolute_error(y_valid, y_pred)
print(f'MAE: {mae}')
rmse = math.sqrt(mse)
print(f'RMSE: {rmse}')
msle = mean_squared_log_error(y_valid, y_pred)
print(f'MSLE: {msle}')
m = math.sqrt(mean_absolute_error(np.log(y_valid), np.log(y_pred)))
print(f'M:{m}')
print(f'Score: {math.sqrt(msle)}')


# # Approfondissement de l'EDA

# ## Colonnes numériques

# Après vérification des colinéarités afin de réduire le nombre de dimensions (colonnes numériques):
# On peut drop : ['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF']
X_full_final = X_full.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1)


# Séléction des colonnes numériques
X_num = X_full_final.select_dtypes(include = ['int','float'])
df_info(X_num)


numeric_columns = X_num.columns
numeric_columns


# Vérification de l'impact des outliers
X_num.describe()


# ### Gestion des nulls : 
# Après vérification des colonnes ayant des nulls, 'LotFrontage' et 'MasVnrArea', on peut les remplir avec la mediane qui est plus appropriée dans la mesure où nous avons des outliers qui influencent la moyenne (surtout pour 'MasVnrArea').
# 
# --> SimpleImputer avec median

# ## Colonnes catégorielles

# Sélection des colonnes catégorielles ordinales et nominales
X_cat = X_full_final.select_dtypes(include = ['category'])
ordinal_cat_columns = ['LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                       'BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond',
                       'PavedDrive','PoolQC','Fence','OverallQual','OverallCond']
cat_columns = ['MSZoning','Alley','Street','LotShape','LandContour','Utilities','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle',
               'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','Electrical',
               'Functional','GarageType','MiscFeature','SaleType','SaleCondition','MSSubClass']
# Tri des colonnes ayant des nulls
nan_is_no = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
             'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df_info(X_cat)
#ordinal et nominal => 2 pipelines différents à voir


pd.set_option('max_columns', None)
X_cat.describe(include='category')

# Remplace NA in 'Electrical' par la valeur la plus fréquente : 
X_full_final['Electrical'] = X_full_final['Electrical'].fillna('SBrkr')


cat_global_columns = X_cat.columns
cat_global_columns


# ### Gestion des nulls :
# 
# Sur les 16 colonnes ayant des nulls, une seule ('Electrical') nécessite un imputer avec le most frequent, pour les autres celà correspond au fait qu'il n'y a pas par exemple de cloture, ou pas de garage ect, dant ce cas là on utilisera un simple imputer avec une valeur donnée
# 
# ### Encodage
# 
# On a séparé les colonnes ordinales pour pouvoir les encoder avec OrdinalEncoder, pour les autres ce sera un OneHot

# # Premières améliorations du modèle (gestion des nulls et encodage ordinal/nominal)


# Définition de la target et des features
y_final = X_full_final.SalePrice
X_final = X_full_final.drop(['SalePrice'], axis=1)


# Séparation en jeu de train et de valid
X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(X_final, y_final, train_size=0.8, test_size=0.2, random_state=0)


### Pipeline final
numeric_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

categorical_encoder = Pipeline(steps=[
    ('const_imputer', SimpleImputer(strategy='constant',fill_value='Abs')),
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))])

ordinal_encoder = Pipeline(steps=[
    ('const_imputer', SimpleImputer(strategy='constant',fill_value='Abs')),
    ('ord_encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipe, selector(dtype_exclude=["category"])),
    ('ord_enc', ordinal_encoder, ordinal_cat_columns),
    ('cat_enc', categorical_encoder, cat_columns)],remainder='passthrough')

reg_final = Pipeline(steps=[('preprocessor', preprocessor),
                            #('PCA', PCA(n_components=18)),
                            ('regressor', RandomForestRegressor())])



reg_final.fit(X_train_final, y_train_final)



# Metrics finaux
print(reg_final.score(X_train_final, y_train_final), reg_final.score(X_valid_final, y_valid_final))
y_pred_final = reg_final.predict(X_valid_final)
mse = mean_squared_error(y_valid_final, y_pred_final)
print(f'MSE: {mse}')
mae = mean_absolute_error(y_valid_final, y_pred_final)
print(f'MAE: {mae}')
rmse = math.sqrt(mse)
print(f'RMSE: {rmse}')
msle = mean_squared_log_error(y_valid_final, y_pred_final)
print(f'MSLE: {msle}')
m = math.sqrt(mean_absolute_error(np.log(y_valid_final), np.log(y_pred_final)))
print(f'M:{m}')
print(f'Score: {math.sqrt(msle)}')


# # Prédictions premier modèle

for col in X_test_full.select_dtypes(include = ['object']):
    X_test_full[col] = X_test_full[col].astype('category')
for col in ['MSSubClass','OverallQual','OverallCond']:
    X_test_full[col] = X_test_full[col].astype('category')

X_test_full['Electrical'] = X_test_full['Electrical'].fillna(value='SBrkr')


y_pred_test = reg.predict(X_test_full)



# Enregistrer les prédictions pour soumission
#output = pd.DataFrame({'Id': X_test_full_final.index,
#                       'SalePrice': y_pred_test})
#output.to_csv('submission.csv', index=False)


# # Prédictions second modèle
X_test_full_final = X_test_full.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1)
y_pred_test_final = reg_final.predict(X_test_full_final)


# Enregistrer les prédictions pour soumission
#output = pd.DataFrame({'Id': X_test_full_final.index,
#                       'SalePrice': y_pred_test_final})
#output.to_csv('submission.csv', index=False)

