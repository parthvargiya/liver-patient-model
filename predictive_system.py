import numpy as np
import pickle

# LOADING THE SAVED MODEL
loaded_model = pickle.load(open('C:/Users/Dell/Desktop/New folder/trained_model.sav', 'rb'))

scaler = pickle.load(open('C:/Users/Dell/Desktop/New folder/scaler.sav', 'rb'))

input_data = (0, 65, 0.7,	0.1,	187,	16,	18,	6.8,	3.3,	0.90)

# CHANGING THE INPUT DATA TO A NUMPY ARRAY
input_data_as_numpy_array = np.asarray(input_data)

# RESHAPE THE NUMPY ARRAY
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# STANDARDIZATION OF THE DATA
std_data = scaler.transform(input_data_reshaped)

prediction = loaded_model.predict(std_data)
print(prediction)

if prediction[0] == 1:
  print("THE PERSON IS A LIVER PATIENT")
else:
  print("THE PERSON IS NOT A LIVER PATIENT")