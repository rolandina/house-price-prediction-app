import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from model.data import Data 


def app():
    st.title('Data Analysis')
    st.markdown("### Predict house prices in USA")

    #prepare cache data
    @st.cache
    def load_data():
        data = Data()
        test = data.get_prepared_test_data()
        train = data.get_prepared_train_data()
        return (train, test)

    train, test = load_data()
    target = "SalePrice"

    plot_type = st.selectbox("Plot type:",("Numerical", "Categorical","Correlation-Resuduals"))

    def create_list_numeric_columns(df):
        return [column for column in df.columns if df[column].dtypes == "float" or (len(df[column].unique())>=15 and df[column].dtypes == "int")]

    def create_list_categoric_columns(df):
        return [column for column in df.columns if df[column].dtypes != "float" and df[column].dtypes != "int"] + [column for column in df.columns if df[column].dtypes == "int" and len(df[column].unique())<15]


    def plot_numeric_features(df):
        numerical_features = create_list_numeric_columns(df)
        sns.set()  # Setting seaborn as default style even if use only matplotlib
        sns.set_palette("Paired")  # set color palette
        fig, axes = plt.subplots(nrows=len(numerical_features),
                                ncols=2,
                                figsize=(10, 3* len(numerical_features)))
        for i, feature in enumerate(numerical_features):
            sns.histplot(data=df, x=feature, kde=True, ax=axes[i, 0])
            sns.boxplot(data=df, x=feature, ax=axes[i, 1])
        st.pyplot(fig)


    def plot_categorical_features(df, target):  
        categorical_features = create_list_categoric_columns(df)

        fig, axes = plt.subplots(nrows=len(categorical_features),
                                ncols=1,
                                figsize=(14, 4* len(categorical_features)))
        for i, feature in enumerate(categorical_features):
            
            df_group = df[[feature, target]].groupby(feature).count()
            #print(df_group)
            sns.boxplot(data=df, x=feature, y= target, ax=axes[i])
            sns.swarmplot(data=df, x=feature, y= target, ax=axes[i], color=".25", size = 2) #swarmplots
            if len(df[feature].unique())>20:
                plt.xticks(rotation=45)
            axes[i].set_title(f"{target} by {feature}")
        st.pyplot(fig)


    def plot_res_corr(df, target): 
        numerical_features = create_list_numeric_columns(df)
        n = len(numerical_features)-1

        fig, axes = plt.subplots(nrows=n,
                                ncols=2,
                                figsize=(14, 4*n))
        i = 0
        for f in numerical_features: 
            if f  != target:
                    sns.regplot(data=df, x=f, y=target, color='blue', ax=axes[i, 0])
                    sns.residplot(data=df, x=f, y=target, color='red', ax = axes[i, 1])
                    i+=1

        st.pyplot(fig)


    if plot_type == "Numerical":
        plot_numeric_features(train)
    elif plot_type == "Categorical":
        plot_categorical_features(train, target)
    elif plot_type == "Correlation-Resuduals":
        plot_res_corr(train, target)


    st.write(train)
    # model_type = st.selectbox("Select Regression Model:", ("Lasso","Ridge", "Linear Regression"))


