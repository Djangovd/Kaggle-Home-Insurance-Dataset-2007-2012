# Kaggle Home Insurance Dataset 2007-2012

In this project I have explored the Home Insurance dataset for 2007-2012 (found here: https://www.kaggle.com/datasets/ycanario/home-insurance?resource=download). 

## Motivation
The dataset contains home insurance policy data, including active and inactive (lapsed or cancelled). I haev derived some insights into the data, and conducted an investigation into the various features and if they can be used to predict if a policy is active or inactive.

## Requirements
All code has been developed using python version 3.10.11.

## Material
### Investigation and Insights
- Jupyter Notebook `./Kaggle_Home_Insurance_Data_2007_2012.ipynb`, which contains the sequential flow of all the investigations that have been carried out
- The data can be found in the `data` subfolder
- Auxiliary functions are located in the `src` subfolder, and they are loaded in the beginning of the Jupyter Notebook

#### Outline of the Jupyter Notebook (roughly)
- Data ingestion
- Data inspection
    - Some insights
- Data transformations and cleansing
- Modeling
    - XGBoost
    - Random Forest Classifier
        - Random Forest Classifier with PCA
        - Random Forest Classifier-based feature importance
    - Random Forest Classifier revisited (with a selection of the most important features)
- Results and Discussion

### MVP / prototype
An example of a more user-friendly prototype has also been created. I used streamlit to create an app that runs one of the best/better models. The source for this can be found
- in the file `app.py` 
In order to run the app, open a browser and type the following in a terminal:
```
$ streamlit run app.py
```
The way the app works:
- In order to make the experience somewhat user friendly, teh user can only select values for the most important features. The remaining features are randomly selected from values in the data
- Once a prediction has been made a table will appear where the entire input fed into the model is visible for inspection

