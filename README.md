# wafer-defect-detection-and-AI-diagnosis
An End-to-End Wafer defect detection and classification prototype with AI-assisted diagnostics

Project Report: https://docs.google.com/document/d/1yX7YYs442Cnwx51rdl6zzrNE8yArN0am6SBih1FNJnU/edit?usp=sharing

This repository contains the code for a prototype Fault Detection and Classification (FDC) system. It detects wafer defects using a custom CNN and generates actionable maintenance reports using the Gemini 2.5 Flash LLM. For detailed information on the methodology, dataset, and results, please refer to the project report linked above.

# How to Run the Code

To run this project on your local machine, follow these steps.

1. Clone the Repository

  Download the project files to your local system.


2. Install Dependencies

  You will need Python installed. Install the required libraries using pip:

  pip install streamlit torch pandas numpy opencv-python matplotlib langchain-google-genai scikit-learn jupyter


3. Dataset Setup

  Due to size limits, the raw dataset is not included in this repository.

  Download the WM-811K dataset (file named LSWMD.pkl) from Kaggle.

  Place the LSWMD.pkl file inside the data/ folder.


4. Retrain the Model - optional

  If you want to process the data and train the model from scratch:

  Preprocess the Data: Run the data handling script to generate the training/testing pickles.

  python 2_DHandPP.py

  Train the CNN: Run the training script to save a new cnn_model.pth.

  python 3_TrainingCNN.py


5. Run the Dashboard

  To launch the AI-assisted diagnosis tool:

  Get your Gemini API Key from Google AI Studio.

  Run the Streamlit application:

streamlit run LLM-RAG.py

Enter your API key in the sidebar and click "Scan Random Wafer" to test the system.

