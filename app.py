import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb')) 
dataset= pd.read_csv('PCA and NN dataset22.csv')

# Extracting independent variable:
x = dataset.iloc[:,0:8].values
# Extracting dependent variable:
y = dataset.iloc[:, -1].values

#Taking Care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(x[:, 1:7 ]) 
#Replacing missing data with the calculated mean value  
x[:, 1:7 ]= imputer.transform(x[:, 1:7 ])  


#For Column Gender 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Female', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(x[:, 0:1]) 
#Replacing missing data with the calculated mean value  
x[:, 0:1]= imputer.transform(x[:, 0:1])  


# Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# Splitting the Dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)


from sklearn.decomposition import PCA
pca= PCA(n_components= None)
x_train= pca.fit_transform(x_train)
x_test= pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_

from sklearn.decomposition import PCA
pca= PCA(n_components= 5)
x_train= pca.fit_transform(x_train)
x_test= pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_



def predict_note_authentication(Gender,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age):

  if(Gender=='Male'):
      ge=1
  else:
      ge=0


  output= model.predict(sc.transform([[ge,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("Person will ",output)
  if output==[0]:
    prediction="Person is Diagnosed"


  if output==[1]:
    prediction="Person is not Diagnosed"


  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">End Term</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Laveena Jethani</p></center>
   <center><p style="font-size:25px;color:white;margin-top:0px;">Sec:B , PIET18CS080 </p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Person Diagnose")
    Gender = st.selectbox('Insert Gender', ('Male', 'Female'))
    Glucose = st.number_input('Insert Glucose level')
    bp = st.number_input('Insert BP level')
    SkinThickness = st.number_input('Insert SkinThickness')
    Insulin = st.number_input('Insert Insulin level')
    BMI = st.number_input('Enter BMI')
    PedigreeFunction = st.number_input('Enter PedigreeFxn')
    Age = st.number_input('Enter Age')

    result=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted that -> {}'.format(result))
    if st.button("About"):
      st.subheader("Laveena Jethani")
      st.subheader("Computer Science,PIET")

if __name__=='__main__':
  main()
