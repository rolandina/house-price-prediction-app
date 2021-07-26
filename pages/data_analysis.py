import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model.data import Data
import matplotlib.gridspec as gridspec
import numpy as np

sns.color_palette("bright")
pal = sns.color_palette("bright").as_hex()

def app():
    st.title('Data Analysis')
    st.markdown("### Predict house prices in USA")
    
    #prepare cache data
    @st.cache
    def load_data():
        return Data()

    data = load_data()

    test = data.get_test_df()
    train = data.get_train_df()
    target = "SalePrice"

    empty, col1 = st.beta_columns([5,1])
    with col1:
        plot_type = st.selectbox("Plot type:",("Numerical", "Categorical","Correlation-Resuduals"))


    def plot_numeric_features(df, max_num_cat = 25):
             
        """Parameters:
        plot_numerical_features(df, max_num_cat = 20)
        df     - data frame with different types of features with target,
        max_num_cat - number of classes in columns where the type is int by default = 25
        
        Output: display plots"""
        
        #condition for categorical features is the same as in plot_categoric_features()
        cat_features = [column for column in df.columns if df[column].dtypes == "object" ] + \
        [column for column in df.columns if df[column].dtypes == "int" and len(df[column].unique())< max_num_cat]
        num_features = [col for col in df.columns if col not in cat_features]
        
        if len(num_features)>0:

            # gridspec inside gridspec
            num_plots = len(num_features)
            #you can change number of columns
            cols = 4 
            rows = num_plots//cols if num_plots%cols == 0 else num_plots//cols +1

            fig = plt.figure(figsize = (cols*5,rows*3))
            gs0 = gridspec.GridSpec(rows, cols, figure=fig)
            for i,feature in enumerate(num_features):
                gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs0[i])

                ax1 = fig.add_subplot(gs00[0, :])
                ax2 = fig.add_subplot(gs00[1:-1, :])

                sns.boxplot(data=df, x=feature, ax=ax1)
                sns.histplot(data=df, x=feature, kde=True, ax=ax2)

                #plt.subplots_adjust(wspace=0, hspace=0.1)
                ax1.set(xlabel='')
                ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
                ax1.axis('off')
            st.pyplot(fig)
                
        else:
            st.write('There are no numerical features in your dataset')


    def plot_categorical_features(df, target):  
        """Parameters:
        df - data frame with different types of features with target,
        target - name of the column of dependent feature
        Output: display plots"""

        # if target is categorical

        #create df of categorical features
        ## Attention on condition
        case = 0  # target is numerical by default
        cat_features = [
            column for column in df.columns if df[column].dtypes == "object"
        ] + [
            column for column in df.columns
            if df[column].dtypes == "int" and len(df[column].unique()) < 20
        ]

        if target in cat_features:
            case = 1
            cat_features.remove(target)

        df_cat = df[cat_features]

        if len(cat_features) > 0:

            # gridspec inside gridspec
            num_plots = len(cat_features)
            #you can change number of columns
            cols = 4
            rows = num_plots // cols if num_plots % cols == 0 else num_plots // cols + 1

            fig = plt.figure(figsize=(cols * 5, rows * 3))
            gs0 = gridspec.GridSpec(rows, cols, figure=fig)
            for i, feature in enumerate(cat_features):
                gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs0[i])

                ax1 = fig.add_subplot(gs00[:-1, :])

                if case == 1:
                    data_dict = {}
                    for target_class in list(df[target].unique()):
                        data_dict[target_class] = []
                        df_target = df[df[target] == target_class]
                        for feature_category in list(df_cat[feature].unique()):
                            df_temp = df_target[df_target[feature] ==
                                                feature_category]
                            data_dict[target_class].append(len(df_temp))
                    plotdata = pd.DataFrame(data_dict,
                                            index=list(df_cat[feature].unique()))
                    plotdata.plot(kind="bar", stacked=True, color=pal, ax=ax1)
                    ax1.set_xlabel(f"Classes in {feature}")
                    ax1.set_ylabel("frequency")
                    for tick in ax1.get_xticklabels():
                        tick.set_rotation(0)
                    

                else:
                    sns.boxplot(data=df, x=feature, y=target, ax=ax1)
                    sns.stripplot(data=df,
                                x=feature,
                                y=target,
                                ax=ax1,
                                color=".25",
                                size=2)  #swarmplots
                    ax1.set_title(f"{target} by {feature}")
            st.pyplot(fig)

        else:
            st.write('There are no categorical features in your dataset')    



    def plot_res_corr(df, target): 


        #condition for categorical features is the same as in plot_categoric_features()
        num_features = [column for column in df.columns if df[column].dtypes != "object" ] 

        if len(num_features)-1>0:

            # gridspec inside gridspec
            num_plots = len(num_features)-1
            #you can change number of columns
            cols = 4 
            rows = num_plots//cols if num_plots%cols == 0 else num_plots//cols +1

            fig = plt.figure(figsize = (cols*5,rows*3))
            gs0 = gridspec.GridSpec(rows, cols, figure=fig)
            for i,f in enumerate(num_features):
                gs00 = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs0[i])

                ax1 = fig.add_subplot(gs00[0:2, :-1])
                ax2 = fig.add_subplot(gs00[2:4, :-1])

                sns.regplot(data=df, x=f, y=target, color=pal[1], ax=ax1)
                sns.residplot(data=df, x=f, y=target, color=pal[3], ax = ax2)

                ax1.set(xlabel='')
                ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

                #plt.subplots_adjust(wspace=0, hspace=0.1)

            st.pyplot(fig)
                
        else:
            st.write('There are no numerical features in your dataset')




    if plot_type == "Numerical":
        plot_numeric_features(train)
    elif plot_type == "Categorical":
        plot_categorical_features(train, target)
    elif plot_type == "Correlation-Resuduals":
        plot_res_corr(train, target)


    st.write(train)
    # model_type = st.selectbox("Select Regression Model:", ("Lasso","Ridge", "Linear Regression"))


