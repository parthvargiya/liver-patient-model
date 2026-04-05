import numpy as np 
import pickle
import streamlit as st
import pandas as pd

# LOADING THE SAVED MODEL
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# LOADING THE SAVED SCALER
scaler = pickle.load(open('scaler.sav', 'rb'))

# CREATING A FUNCTION FOR PREDICTION 
def liverpatient_prediction(input_data):

  # CHANGING THE INPUT DATA TO A NUMPY ARRAY
  input_data_as_numpy_array = np.asarray(input_data)

  # RESHAPE THE NUMPY ARRAY
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  # STANDARDIZATION OF THE DATA
  # UPDATED COLUMN NAMES TO MATCH YOUR ERROR MESSAGE EXACTLY
  cols = [
      'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
      'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
      'Aspartate_Aminotransferase', 'Total_Protiens', 
      'Albumin', 'Albumin_and_Globulin_Ratio'
  ]
  
  input_df = pd.DataFrame(input_data_reshaped, columns=cols)
  std_data = scaler.transform(input_df)

  prediction = loaded_model.predict(std_data)

  if prediction[0] == 1:
    return "THE PERSON IS A LIVER PATIENT"
  else:
    return "THE PERSON IS NOT A LIVER PATIENT"
  

def main():

  # GIVING A TITLE 
  st.title("Liver Patient Disease Prediction Webpage")

  # GETTING THE INPUT DATA FROM THE USER 
  Age = st.text_input('ENTER THE AGE :')
  Gender = st.text_input('GENDER : ')
  Total_Bilirubin = st.text_input('TOTAL BILIRUBIN VALUE :')
  Direct_Bilirubin = st.text_input('DIRECT BILIRUBIN VALUE :')
  Alkaline_Phosphotase = st.text_input('ALKALINE PHOSPHOTASE :')
  Alamine_Aminotransferase = st.text_input('ALAMINE AMINOTRANSFERASE :')
  Aspartate_Aminotransferase = st.text_input('ASPARTATE AMINOTRANSFERASE :')
  Total_Protiens = st.text_input('TOTAL PROTEINS :')
  Albumin = st.text_input('ALBUMIN :')
  Albumin_and_Globulin_Ratio = st.text_input('ALBUMIN AND GLOBULIN RATIO :')

  # CODE FOR PREDICTION 
  diagnosis = ''

  # CREATING A BUTTON FOR PREDICTION
  if st.button('Test Result'):
    # CHECK IF ALL FIELDS ARE FILLED TO AVOID EMPTY STRING ERROR
    try:
        user_input = [
            float(Age), float(Gender), float(Total_Bilirubin), float(Direct_Bilirubin), 
            float(Alkaline_Phosphotase), float(Alamine_Aminotransferase), 
            float(Aspartate_Aminotransferase), float(Total_Protiens), 
            float(Albumin), float(Albumin_and_Globulin_Ratio)
        ]
        
        diagnosis = liverpatient_prediction(user_input)
        st.success(diagnosis)
    except ValueError:
        st.error("PLEASE ENTER VALID NUMBERS IN ALL FIELDS")

if __name__ == '__main__':
  main()
