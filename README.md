# Multi-Disease Prediction System

This project is a web-based system for predicting multiple diseases using both Machine Learning (ML) and Deep Learning (DL) techniques. It offers a user-friendly platform for early and accurate disease diagnosis. The system currently supports predictions for diabetes, heart disease, pneumonia, and brain tumors.

## Features

- **Diabetes Prediction:** Uses Support Vector Machine (SVM) with numerical health data.
- **Heart Disease Prediction:** Utilizes a Decision Tree classifier with structured health data.
- **Pneumonia Detection:** Employs a custom Convolutional Neural Network (CNN) to classify chest X-ray images.
- **Brain Tumor Classification:** Implements a custom CNN to classify MRI images into four categories.

## Motivation

This project bridges the gap in multi-disease prediction systems by integrating diverse ML and DL models into a single platform. It addresses the need for scalable, real-time solutions for early disease detection and improves accessibility to diagnostic tools in resource-limited regions.

## Methodology

### Diabetes Prediction
- **Model:** SVM with Radial Basis Function (RBF) Kernel
- **Dataset:** Pima Indians Diabetes Dataset (Kaggle)
- **Accuracy:** 76%

### Heart Disease Prediction
- **Model:** Decision Tree
- **Dataset:** Cleveland Heart Disease Dataset
- **Accuracy:** 88%

### Pneumonia Detection
- **Model:** Custom CNN
- **Dataset:** Chest X-ray dataset with pneumonia labels
- **Accuracy:** 90%

### Brain Tumor Classification
- **Model:** Custom CNN
- **Dataset:** Brain tumor dataset
- **Accuracy:** 91%

## Experimentation Setup
- **Preprocessing:** Includes normalization, standardization, and data augmentation for image datasets.
- **Training:** Models trained on labeled datasets using techniques like backpropagation for CNNs.
- **Prediction:** Provides real-time predictions based on user inputs (numerical data or images).

## Web Application

The system features a Streamlit-based interface for seamless interaction:
- Home page for navigation.
- Input forms for numerical data or uploading medical images.
- Real-time results displayed with high accuracy.

### Live Demo
- **Web App:** [Multi-Disease Prediction System](https://multiplediseasepredictionbycrn.streamlit.app/)

### Code Repository
- [GitHub Repository](https://github.com/chetan-narware/Multiple_Disease_prediction)

## Results Comparison

| Model               | Proposed Accuracy (%) | Other Research Accuracy (%) |
|---------------------|-----------------------|-----------------------------|
| Diabetes Prediction | 76                   | 74                          |
| Heart Disease       | 88                   | 85                          |
| Pneumonia           | 90                   | 87                          |
| Brain Tumor         | 91                   | 89                          |

## Conclusion

This system is a step forward in making healthcare diagnosis more accessible and efficient. By integrating advanced ML and DL techniques, it offers a reliable and scalable solution for early disease detection. Future enhancements could include expanding the range of diseases and incorporating more advanced models.

