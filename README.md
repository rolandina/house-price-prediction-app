# *HOUSE PRICE PREDICTION APP* 

## *What it does?*
House price prediction application is an application which were made for educational purposes in order to obrain new skills (Fast API and streamlit library)


*House price prediction application consists from files:*


- [app.py](https://github.com/rolandina/house-price-prediction-app/blob/master/app.py) - main page of the app and connector to other pages
- [multipage.py](https://github.com/rolandina/house-price-prediction-app/blob/master/multipage.py) - file with MultiPage class which allow us to create new page automatically

In the folder [model](https://github.com/rolandina/house-price-prediction-app/tree/master/model):
- [data.py](https://github.com/rolandina/house-price-prediction-app/blob/master/model/data.py) - file with class Data() which get data from fastapi server. Original data is taken from here

In the folder [data](https://github.com/rolandina/house-price-prediction-app/tree/master/data):
- [data/data_description.txt](https://github.com/rolandina/house-price-prediction/blob/master/data/data_description.txt)
- [data/train.csv](https://github.com/rolandina/house-price-prediction-app/blob/master/data/train.csv)
- [data/test.csv](https://github.com/rolandina/house-price-prediction-app/blob/master/data/test.csv)

In folder [pages](https://github.com/rolandina/house-price-prediction-app/tree/master/pages)
- [description.py]()
- [data_analysis.py](https://github.com/rolandina/house-price-prediction-app/blob/master/pages/data_analysis.py)
- [model_and_prediction.py](https://github.com/rolandina/house-price-prediction-app/blob/master/pages/model_and_prediction.py)


- [requirements.txt](https://github.com/rolandina/house-price-prediction-app/requirements.txt) -  file with prerequisites libraries which heroku need in oder to create server


## General Prerequisites (for running and building)

* [Anaconda](https://www.anaconda.com/products/individual)
* [GitHub](https://github.com)

After installing anaconda and python you have to set your environment with environment.yml file

```bash
conda env create -f environment.yml
conda activate house_predict
```

## To build and run


```
# Clone this repository 
git clone https://github.com/rolandina/house-price-prediction-git.git
```


## Contributing

This application was created with following tools:
- [heroku](https://dashboard.heroku.com/apps)
- [FastApi](https://fastapi.tiangolo.com/)
- [streamlit](https://streamlit.io/)

The data for this project was taken from [Kaggle Data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
If you have any questions, please reach me at ms.nina.smirnova@gmail.com

