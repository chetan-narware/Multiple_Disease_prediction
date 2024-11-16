import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image as im # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image


heart_model = pickle.load(open("./models/heart.sav",'rb'))
parkinsons_model = pickle.load(open("./models/parkinsons.sav",'rb'))
diabeties_model = pickle.load(open("./models/diabetes.sav",'rb'))
pm = load_model("./models/pneumonia_model.h5")
bm = load_model("./models/brain_model.h5")
loaded_scaler = pickle.load(open('./models/sc_diabeties.pkl','rb'))

def diabetes_prediction(input_data):
    inp = np.asarray(input_data)
    inp_reshaped = inp.reshape(1,-1)
    inp_reshaped = loaded_scaler.transform(inp_reshaped)
    pred = diabeties_model.predict(inp_reshaped)

    if pred[0] == 0:
        return 'Non Diabetic'
    else:
        return 'Diabetic'
    
def  Heart_disease_prediction(input_data):
    inp = np.asarray(input_data)
    inp_reshaped = inp.reshape(1,-1)
    pred = heart_model.predict(inp_reshaped)

    if pred[0] == 0:
        return 'Have A Heart Problem'
    else:
        return 'Do Not Have A Heart Problem'
    
def predict_Pneumonia(img):
    img = img.convert("RGB")
    img_array = im.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = pm.predict(img_array)
    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
def predict_brain(img):
    img_array = im.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = bm.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home','Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Brain Tumor Classification',
                           'Pneumonia Prediction'],default_index= 0)
    
    

if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction Using SVM')
    
    #getting input data from the use
    pregnancies = float(st.text_input("Number of Pregnancies", "1"))
    glucose = float(st.text_input("Glucose Level (mg/dL)", "100.0"))
    blood_pressure = float(st.text_input("Blood Pressure (mm Hg)", "70.0"))
    skin_thickness = float(st.text_input("Skin Thickness (mm)", "20.0"))
    insulin = float(st.text_input("Insulin Level (µU/mL)", "80.0"))
    bmi = float(st.text_input("BMI (Body Mass Index)", "25.0"))
    diabetes_pedigree_function = float(st.text_input("Diabetes Pedigree Function", "0.5"))
    age = float(st.text_input("Age", "25"))

    # Button to submit
    des = ''
    if st.button("Test Results"):
        des = diabetes_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin,
        bmi, diabetes_pedigree_function, age])
    
    st.success(des)

if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction Using Descition Tree')
    #input
    age = int(st.text_input("Age", "25"))
    sex = int(st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0]))
    cp = int(st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3]))
    trestbps = int(st.text_input("Resting Blood Pressure", "120"))
    chol = int(st.text_input("Cholesterol Level", "200"))
    fbs = int(st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0]))
    restecg = int(st.selectbox("Resting ECG Results (0-2)", [0, 1, 2]))
    thalach = int(st.text_input("Maximum Heart Rate Achieved", "150"))
    exang = int(st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0]))
    oldpeak = float(st.text_input("ST Depression Induced by Exercise", "1.0"))
    slope = int(st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2]))
    ca = int(st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3]))
    thal = int(st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2]))


    # Button to submit
    des = ''
    if st.button("Test Results"):
        des = Heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
    st.success(des)

    

if selected == 'Brain Tumor Classification':

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Checking if an image file has been uploaded
    if uploaded_file is not None:
        st.write('image uploaded succesfully')
        image = Image.open(uploaded_file)
        des = ''
        resized_image = image.resize((150,150))
        if st.button('Brain Tumor Type'):
            des = predict_brain(resized_image)
        st.success(des)

if selected == 'Pneumonia Prediction':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Checking if an image file has been uploaded
    if uploaded_file is not None:
        st.write('image uploaded succesfully')
        image = Image.open(uploaded_file)
        des = ''
        resized_image = image.resize((150,150))
        if st.button('Result'):
            des = predict_Pneumonia(resized_image)
        st.success(des)

# Home page content
if selected == 'Home':
    st.title("Welcome to the Disease Prediction System")
    st.markdown("""
    This is a web application for the prediction of several diseases using machine learning and deep learning models. 
    The application can predict the following:

    - **Diabetes**: Using SVM classifier based on health parameters.
    - **Heart Disease**: Using a Decision Tree classifier.
    - **Parkinson's Disease**: Using SVM based on voice features.
    - **Brain Tumor Classification**: Using a CNN model to classify types of brain tumors.
    - **Pneumonia Prediction**: Using a CNN model to predict pneumonia from chest X-ray images.

    ## How to use:
    1. Choose a disease prediction from the sidebar.
    2. For disease predictions based on input features, fill in the required fields.
    3. For image-based predictions (brain tumor or pneumonia), upload an image and click the respective button to get the results.

    Let's get started by selecting one of the prediction options from the menu!
    """)
