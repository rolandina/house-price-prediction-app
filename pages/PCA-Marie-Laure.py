

X_test = X_full_final.select_dtypes(exclude=["category"])

# Définition de la target et des features
y_pca = X_test.SalePrice
X_pca = X_test.drop(['SalePrice'], axis=1)

# Séparation en jeu de train et de valid
X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(X_pca, y_pca, train_size=0.8, test_size=0.2,
                                                      random_state=0)



# imputer données manquantes sur le train:
num_imputer = SimpleImputer(strategy="median")
X_train_num_imputed = num_imputer.fit_transform(X_train_pca)
X_train_num_imputed_df = pd.DataFrame(X_train_num_imputed, columns = X_train_pca.columns)


# StandardScaler sur le train:
num_scaler = StandardScaler()
X_train_num_scaled = num_scaler.fit_transform(X_train_num_imputed)
X_train_num_scaled_df = pd.DataFrame(X_train_num_scaled, columns = X_train_pca.columns)
X_train_num_scaled_df.shape


# ## Identification du nombre de dimensions permettant de contribuer à 95% de la variance des features
# - Matrice de covariance des features : vecteurs propres correspondent aux composantes principales, valeurs propres correspondent au % de variance du modèle contenu dans la composante principale.
# - **pca.explained_variance_ratio_** renvoie un array de ces valeurs propres. On peut en faire la somme cumulative pour voir l'évolution de la proportion de variance totale expliquée par le modèle au fur et à mesure de l'ajout des dimensions.
# - on peut ploter cette somme cumulative de la pca_explained_variance_ratio pour identifier le nombre oprimal de dimensions (au moment où on a une cassure sur la courbe)
# 


pca = PCA() # on instancie sans fixer le n_components
pca.fit(X_train_num_scaled_df)
# % de variance du modèle contenu dans chaque composante principale
pca.explained_variance_ratio_


print("la première composante principale explique:", int(round(pca.explained_variance_ratio_[0],2)*100),"% de la variance")
print("la seconde composante principale explique:", int(round(pca.explained_variance_ratio_[1],2)*100),"% de la variance")

cumsum = np.cumsum(pca.explained_variance_ratio_)
cumsum


d = np.argmax(cumsum >= 0.95) + 1
d # nombre de dimensions pour lesquelles la variance est au total de plus de 95%


# représentation graphique de CUMSUM :
# code du livre d'A Géron
plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, X_train_num_scaled_df.shape[1], 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
#plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             #arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
plt.show()


# ## Projection de X sur 23 dimensions


# application 
pca = PCA(n_components = d)
X_reduced = pca.fit_transform(X_train_num_scaled_df)
X_reduced.shape


# ## Modèle

# Pipeline pca
numeric_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
    
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipe, selector(dtype_exclude=["category"]))])

reg_pca = Pipeline(steps=[('preprocessor', preprocessor),
                          ('PCA', PCA(n_components=23)),
                          ('regressor', RandomForestRegressor())])



# Visualisation de la variance expliquée
X_test_pca = X_pca
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(imputer,scaler,pca)
pipeline.fit(X_test_pca)
features = range(pca.n_components_)
plt.figure(figsize=(20,10))
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


reg_pca.fit(X_train_pca, y_train_pca)


# Metrics finaux
print(reg_pca.score(X_train_pca, y_train_pca), reg_pca.score(X_valid_pca, y_valid_pca))
y_pred_pca = reg_pca.predict(X_valid_pca)
mse = mean_squared_error(y_valid_pca, y_pred_pca)
print(f'MSE: {mse}')
mae = mean_absolute_error(y_valid_pca, y_pred_pca)
print(f'MAE: {mae}')
rmse = math.sqrt(mse)
print(f'RMSE: {rmse}')
msle = mean_squared_log_error(y_valid_pca, y_pred_pca)
print(f'MSLE: {msle}')
m = math.sqrt(mean_absolute_error(np.log(y_valid_pca), np.log(y_pred_pca)))
print(f'M:{m}')
print(f'Score: {math.sqrt(msle)}')


# n_components = 18
# 0.974562708593533 0.7719492340942051
# MSE: 1574884373.5215192
# MAE: 21654.995194063926
# RMSE: 39684.81288253126
# MSLE: 0.026453861295648088
# M:0.3331851088890741
# Score: 0.16264643031941428
# 
# n_compoments = 2
# 0.9699847195678531 0.8285868034922612
# MSE: 1183753816.932704
# MAE: 22397.72172374429
# RMSE: 34405.72360716606
# MSLE: 0.026512309658704138
# M:0.34471873169829437
# Score: 0.1628260103874812
# 
# n_components = 23
# 0.9723819754484854 0.7656924488021736
# MSE: 1618092794.005273
# MAE: 21828.6378173516
# RMSE: 40225.52416072752
# MSLE: 0.026884194056412125
# M:0.3340102387502967
# Score: 0.16396400231883865


from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def lets_try(train,labels):
    results={}
    def test_model(clf):
        
        cv = KFold(n_splits=5,shuffle=True,random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=r2)
        scores=[r2_val_score.mean()]
        return scores

    clf = linear_model.LinearRegression()
    results["Linear"]=test_model(clf)
    
    clf = linear_model.Ridge()
    results["Ridge"]=test_model(clf)
    
    clf = linear_model.BayesianRidge()
    results["Bayesian Ridge"]=test_model(clf)
    
    clf = linear_model.HuberRegressor()
    results["Hubber"]=test_model(clf)
    
    clf = linear_model.Lasso(alpha=1e-4)
    results["Lasso"]=test_model(clf)
    
    clf = BaggingRegressor()
    results["Bagging"]=test_model(clf)
    
    clf = RandomForestRegressor()
    results["RandomForest"]=test_model(clf)
    
    clf = AdaBoostRegressor()
    results["AdaBoost"]=test_model(clf)
    
    clf = svm.SVR()
    results["SVM RBF"]=test_model(clf)
    
    clf = svm.SVR(kernel="linear")
    results["SVM Linear"]=test_model(clf)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["R Square Score"] 
    #results=results.sort(columns=["R Square Score"],ascending=False)
    results.plot(kind="bar",title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([0.5,1])
    return results


X_pca = preprocessor.fit_transform(X_pca, y_pca)
lets_try(X_pca, y_pca)


# # PCA sans SKLearn (calcul matriciel)

# ## Identification des composantes principales

# PCA sur le X numérique - méthode en utilisant numpy
# on prend le X numérique scalé autour de 0 grâce au standard scaler
X_train_num_scaled_array = X_train_num_scaled_df.values
U, s, Vt = np.linalg.svd(X_train_num_scaled_array) # svd = Singular Value Decomposition. 
# svd : c'est l'équation de décomposition de la matrice X qui permet d'en extraire la matrice de composantes principales
c1 = Vt.T[:, 0] # vecteur unitaire définissant la première composante principale
c2 = Vt.T[:, 1] # et la seconde CP
c1


# ## Projection  sur d dimensions

# matrice des deux vecteurs supports des deux premières CP
W2 = Vt.T[:, :2]
# produit scalaire de X avec cette matrice
X2D = X_train_num_scaled_array.dot(W2)
X2D # shape = (1168, 2) => on a réduit notre jeu de données à 2 dimensions



# matrice des deux vecteurs supports des 4 premières CP
W4 = Vt.T[:, :4]
# produit scalaire de X avec cette matrice
X4D = X_train_num_scaled_array.dot(W4)
X4D.shape # shape = (1162, 4) => on a réduit notre jeu de données à 4 dimensions


# # Représentations graphiques de la PCA

# ## Représentation graphique du poids des features dans les composantes principales

pca.components_


# heat-plot to see how the features mixed up to create the components.

plt.figure(figsize=(10,50))
plt.matshow(pca.components_,cmap='viridis')
plt.ylabel("Components")
plt.xlabel("Features")
plt.title("Pondération des features dans chaque composante principale")
#plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=8)
plt.colorbar()
#plt.xticks(range(len(df_main.columns)),df_main.columns,rotation=65,ha='left')
#plt.tight_layout()
plt.show();


# ## Représentation graphique des données selon les composantes principales (obj = identifier des patterns)
# **Composantes principales n°1 et 2**


import plotly.express as px
from sklearn.decomposition import PCA


X = X_train_num_scaled_df

pca = PCA(n_components=23)
components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=y_train, 
                 labels = {
    "0":"Principal Component n°1",
    "1":"Principal Component n°2"
                        },
                title="Projection de X sur les deux premiers axes des Composantes Principales") 
fig.show()

