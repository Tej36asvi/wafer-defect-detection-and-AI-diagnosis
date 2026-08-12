# Wafer Defect Detection and AI Diagnosis

This repository contains a wafer fault detection and classification workflow.
It uses a CNN model for defect classification and a Streamlit dashboard for diagnosis support.

## Project Report

https://docs.google.com/document/d/1yX7YYs442Cnwx51rdl6zzrNE8yArN0am6SBih1FNJnU/edit?usp=sharing

## Repository Files

- `1_InspectingDataset.py`: dataset inspection and sample visualization
- `2_DHandPP.py`: data handling and preprocessing pipeline
- `3_TrainingCNN.py`: CNN training script
- `LLM-RAG.py`: Streamlit dashboard with report generation
- `defect_context.py`: static defect knowledge base

## Setup

Install dependencies:

```bash
pip install streamlit torch pandas numpy opencv-python matplotlib langchain-google-genai scikit-learn jupyter
```

## Dataset

The raw dataset is not included in this repository.

1. Download WM-811K (`LSWMD.pkl`) from Kaggle.
2. Place `LSWMD.pkl` in the `data/` folder.

## Training

1. Preprocess data:

   ```bash
   python 2_DHandPP.py
   ```

2. Train the model:

   ```bash
   python 3_TrainingCNN.py
   ```

## Run the Dashboard

1. Get a Gemini API key from Google AI Studio.
2. Start the app:

   ```bash
   streamlit run LLM-RAG.py
   ```

3. Enter the API key in the sidebar.
4. Click **Scan Random Wafer** to run detection.
