import streamlit as st
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model.data import Data

def app():
    #prepare cache data
    @st.cache
    def load_data():
        return Data()

    data = load_data()
    train = data.get_train_df()
    text_markdown = """


# COURS PCA

### LIENS Utiles
- https://www.kaggle.com/hassanamin/principal-component-analysis-with-code-examples/notebook
- https://www.kaggle.com/vipulgandhi/pca-beginner-s-guide-to-dimensionality-reduction 
- https://www.kaggle.com/miguelangelnieto/pca-and-regression : exemple avec Sales Prices

### Intérêt de la PCA ==> réduire les dimensions du dataset (fléau de la dimension)
- Plus il y a de dimensions moins les prédictions sont robustes, essentiellement en raison d'un **risque d'overfitting** (car plus le nombre de dimensions est important, plus l'éloignement entre les points est grand) ==> difficulté à généraliser
- Plus les features sont nombreux, plus le **temps d'entrainement** des modèles est long
- La PCA accroit donc pas nécessairement l'accuracy (risque de perte d'information). Mais, tout comme la régularisation, permet de maximiser le catactère généralisable d'un modèle (on cherche à minimiser la variance, même si ca peut être au prix d'un biais un peu plus élevé sur le train set). 

### Points de vigilance
- ne fonctionne que sur les **variables quantitatives continues** ==> pas de catégorielles
- ne fonctionne que sur des données centrées autour de 0  ==> utiliser le **StandardScaler**

### Mise en oeuvre de la PCA (théorie)

<br><br>**1) On détermine les vecteurs supports des axes :**
- Le premier axe est celui pour lequel la projection des variables sur cet axe conserve la plus grande part de variance (on peut aussi dire qu'il s'agit de l'axe qui minimise la moyenne du carré des distances entre les données d'origine et leurs projections)
- Ensuite on passe au second axe, orthogonal au premier, qui contribue le plus à la variance résiduelle. Et ainsi de suite jusqu'à ce qu'il ne reste plus de variance résiduelle à expliquer.

<br><br> **2) On détermine le nombre de dimensions :**
- L'objectif est ensuite de déterminer le nombre d'axes à choisir pour conserver (représenter) le max de la variance du dataset (ie perdre le moins d'information) tout en parvenant à l'objectif de réduction de la dimensionnalité. Pour ca on utilise sous sklearn les outils **d'explained_variance_ratio** et la **pca.get_covariance**  

# Décomposition de la PCA avec SKLearn
- **En amont** : imputer les données manquantes et scaler les données
- **Première étape** : identifier le nombre de dimensions (ie le nombre d'axes) permettant de contribuer à expliquer un % donné de la variance des features
- **Deuxième étape** : projeter X sur ces nouveaux axes (c'est ce que fait le PCA.fit_transform) : si on choisit 2 dimensions, on va passer de X.shape=(1200,29) à X.shape=(1200,2) une fois transformé
- **Troisième étape** : appliquer le modèle de ML à X transformé.

## Préprocessing Numérique. 
Objectifs ==> imputer les valeurs manquantes et centrer les données autour de 0"""

    st.markdown(text_markdown, unsafe_allow_html=True)

    target = 'SalePrice'

    train = train.drop(['Id' , 'MSSubClass', 'OverallQual', 'OverallCond'], axis=1)

    X, y = train[[col for col in train.columns if col != target]], train[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    X_train_pca = X_train.select_dtypes(include=['int', 'float'])

    num_imputer = SimpleImputer(strategy="median")
    X_train_num_imputed = num_imputer.fit_transform(X_train_pca)
    X_train_num_imputed_df = pd.DataFrame(X_train_num_imputed, columns = X_train_pca.columns)

    num_scaler = StandardScaler()
    X_train_num_scaled = num_scaler.fit_transform(X_train_num_imputed)
    X_train_num_scaled_df = pd.DataFrame(X_train_num_scaled, columns = X_train_pca.columns)

    pca = PCA()
    pca.fit(X_train_num_scaled_df)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    fig = plt.figure(figsize=(6,4))
    plt.title("Somme cumulative de la variance expliquée")
    plt.plot(cumsum, linewidth=2)
    plt.axis([0, X_train_num_scaled_df.shape[1], 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, 0.95], "k:")
    plt.plot([0, d], [0.95, 0.95], "k:")
    plt.plot(d, 0.95, "ko")
    #plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
                 #arrowprops=dict(arrowstyle="->"), fontsize=16)
    plt.grid(True)
    st.pyplot(fig)

    fig2 = plt.figure(figsize=(20,10))
    features = range(pca.n_components_)
    plt.title('Visualisation de la variance expliquée')
    plt.bar(features, pca.explained_variance_ratio_)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.xticks(features)
    st.pyplot(fig2)
