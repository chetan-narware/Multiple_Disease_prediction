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
Pneumonia_model = load_model('pneumonia_model.h5')
brain_model = load_model('brain_model.h5')
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
    
def parkinsons_prediction(input_data):
    inp = np.asarray(input_data)
    inp_reshaped = inp.reshape(1,-1)
    pred = parkinsons_model.predict(inp_reshaped)

    if pred[0] == 0:
        return 'Have Parkinsons'
    else:
        return 'Do Not Have Parkinsons'
    
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
    prediction = Pneumonia_model.predict(img_array)
    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
def predict_brain(img):
    img_array = im.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = brain_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home','Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Disease Prediction',
                           'Brain Tumor Classification',
                           'Pneumonia Prediction'],default_index= 0)
    
    

if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction Using SVM')
    
    #getting input data from the use
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, value=100.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0.0, max_value=500.0, value=80.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=60.0, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

    # Button to submit
    des = ''
    if st.button("Test Results"):
        des = diabetes_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin,
        bmi, diabetes_pedigree_function, age])
    
    st.success(des)

if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction Using Descition Tree')
    #input
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

    # Button to submit
    des = ''
    if st.button("Test Results"):
        des = Heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
    st.success(des)



if selected == 'Parkinsons Disease Prediction':

    st.title('Parkinsons Disease Prediction Using SVM')
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, value=120.0)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, value=150.0)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, value=100.0)
    mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, value=0.005)
    mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, value=0.00005)
    mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, value=0.003)
    mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, value=0.005)
    jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, value=0.015)
    mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, value=0.03)
    mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, value=0.3)
    shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, value=0.02)
    shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, value=0.025)
    mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, value=0.03)
    shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, value=0.08)
    nhr = st.number_input("NHR", min_value=0.0, value=0.02)
    hnr = st.number_input("HNR", min_value=0.0, value=20.0)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4)
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.6)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0, value=-5.0)
    spread2 = st.number_input("Spread2", min_value=-5.0, max_value=5.0, value=0.5)
    d2 = st.number_input("D2", min_value=0.0, value=2.0)
    ppe = st.number_input("PPE", min_value=0.0, value=0.2)

    # Button for prediction
    des = ''
    if st.button('Predict Parkinson\'s Disease'):
        input_data = np.array([mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, 
                               mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, 
                               mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe])
        des = parkinsons_prediction(input_data)
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
