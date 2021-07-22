import streamlit as st


def app():
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
