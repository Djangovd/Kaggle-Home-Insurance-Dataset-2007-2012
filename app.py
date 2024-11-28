import code
import json
import re
import streamlit as st
import numpy as np
import pandas as pd
import joblib

features = ['CLAIM3YEARS',
  'P1_EMP_STATUS',
  'BUS_USE',
  'NCD_GRANTED_YEARS_C',
  'CONTENTS_COVER',
  'UNSPEC_HRP_PREM',
  'P1_MAR_STATUS',
  'P1_POLICY_REFUSED',
  'P1_SEX',
  'APPR_ALARM',
  'BEDROOMS',
  'ROOF_CONSTRUCTION',
  'WALL_CONSTRUCTION',
  'FLOODING',
  'LISTED',
  'MAX_DAYS_UNOCC',
  'NEIGH_WATCH',
  'OCC_STATUS',
  'OWNERSHIP_TYPE',
  'PAYING_GUESTS',
  'PROP_TYPE',
  'SAFE_INSTALLED',
  'SUBSIDENCE',
  'YEARBUILT',
  'PAYMENT_METHOD',
  'HOME_EM_ADDON_PRE_REN',
  'HOME_EM_ADDON_POST_REN',
  'HP1_ADDON_PRE_REN',
  'HP1_ADDON_POST_REN',
  'HP2_ADDON_PRE_REN',
  'HP2_ADDON_POST_REN',
  'HP3_ADDON_PRE_REN',
  'HP3_ADDON_POST_REN',
  'MTA_FLAG',
  'LAST_ANN_PREM_GROSS',
  'COVER_AGE',
  'P1_AGE']

sub_features = ['LAST_ANN_PREM_GROSS', # The most important features
 'P1_MAR_STATUS',
 'COVER_AGE',
 'PAYMENT_METHOD',
 'P1_AGE',
 'MAX_DAYS_UNOCC',
 'UNSPEC_HRP_PREM',
 'NCD_GRANTED_YEARS_C',
 'PROP_TYPE',
 'CONTENTS_COVER',
 'YEARBUILT',
 'HP2_ADDON_POST_REN',
 'BEDROOMS',
 'MTA_FLAG',
 'HP1_ADDON_POST_REN',
 'HOME_EM_ADDON_PRE_REN',
 'OWNERSHIP_TYPE']


def main():
    """
    Main function for the Streamlit app.

    NB! For features of little importance, random values from the reference dataframe is used. 

    """

    # Load the mapping of categorical values to numerical values
    cat_to_code = json.load(open('./data/cat_to_code.json'))
    code_to_cat = {k: {v: k for k, v in v.items()} for k, v in cat_to_code.items()}

    # Load the trained model (assuming you have saved it as 'model.pkl')
    model = joblib.load('./models/best_model.pkl')

    # Load the reference dataframe
    ref_df = pd.read_pickle('./data/hi_df_cat_2.pkl')

    # Define the feature names (replace with your actual feature names)
    feature_names = features

    st.title('Active/Inactive Home Insurance Policy Prediction')

    # Create input fields for each feature
    input_data = []
    PCA_prefx = "PC_"
    PCA_features = [PCA_prefx+str(i) for i in range(4)]
    for feature in feature_names:
        if feature in sub_features:
            if feature in cat_to_code:
                value = st.selectbox(f'Select value for {feature}', cat_to_code[feature].keys())
                print(f"Value: {value}, code: {cat_to_code[feature][value]}, cat: {code_to_cat[feature]}")
                value = cat_to_code[feature][value]
            else:
                value = st.number_input(f'Enter value for {feature}', value=0)
        else:
            value = ref_df[feature].sample(1) # Use random value from the reference dataframe. This is to save time when running the app and populate the fields with values.
            value = value.values[0]
        input_data.append(value)
        
    print(f"Input data: {input_data}")


    # Predict the output
    if st.button('Predict'):
        prediction = model.predict([input_data])
        if prediction[0] == 1:
            st.success('The prediction is: Active')
        else:
            st.error('The prediction is: Inactive')
        probability = model.predict_proba([input_data])[0] * 100
        st.write(f'Probability of being active: {probability[1]:.2f}%')
        st.write(f'Probability of being inactive: {probability[0]:.2f}%')

    # Display the input data
    st.subheader('The entire input data for this prediction:')
    df = pd.DataFrame([input_data], columns=feature_names)
    st.table(df)


if __name__ == '__main__':
    main()