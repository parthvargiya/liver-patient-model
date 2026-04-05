# Liver Patient Disease Prediction

A Machine Learning web application that predicts whether a person is a liver patient based on clinical blood test parameters.


## Project Overview

This project uses a **Support Vector Machine (SVM)** classifier trained on the Indian Liver Patient Dataset from Kaggle to predict liver disease. The trained model is deployed as an interactive web application using **Streamlit**.


## Repository Structure

File, Description

`indian_liver_patient.csv', Dataset (583 records, 10 features) from Kaggle 
`liver_patient.ipynb`, Google Colab notebook — data cleaning, preprocessing, model training
`trained_model.sav`, Saved SVM model (Pickle)
`scaler.sav`, Saved StandardScaler (Pickle)
`predictive_system.py`, Script to run a single prediction from the command line
`webapp.py`, Streamlit web application for interactive prediction


## Dataset

*Source:* [Indian Liver Patient Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
*Records:* 583 patients
*Features:* Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alamine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin & Globulin Ratio
*Target:* 1 = Liver Patient, 2 = Not a Liver Patient

---

## How It Works

1. Data cleaned and preprocessed in Google Colab
2. Gender encoded (Male = 1, Female = 0)
3. Features normalized using StandardScaler
4. SVM model trained on 80% of data, tested on 20%
5. Model and scaler exported using Pickle
6. Streamlit app loads both `.sav` files for real-time prediction


## Model Performance

Metric:- Value

Algorithm :- Support Vector Machine (SVM)
Train-Test Split :- 80 / 20
Training Accuracy :- ~71%
Testing Accuracy :- ~72%
Best Accuracy Achieved :- ~79%

**Accuracy was limited due to high correlation between features (e.g., Total vs Direct Bilirubin, Total Proteins vs Albumin), a known challenge with biological marker datasets.**


## Running the App

### Install dependencies

pip install numpy pandas scikit-learn streamlit (via terminal)


### Run the Streamlit web app

python-m streamlit run webapp.py (via terminal)


### Run a single prediction

python predictive_system.py (run directly)


## Tech Stack

Python
Google Colaboratory
scikit-learn (SVM, StandardScaler)
Streamlit
Pickle
pandas, numpy


## Project Context

Built as a Capstone Project under the **Microsoft Elevate Internship Program**.
