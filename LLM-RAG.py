import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importing the local knowledge base
from defect_context import DEFECT_KNOWLEDGE_BASE

# Setting up the page layout
st.set_page_config(page_title="FDC System", layout="wide")

# Checking for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

# MODEL ARCHITECTURE 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Classifier
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# LABEL MAP
label_map = {
    0: 'Center',
    1: 'Donut',
    2: 'Edge-Loc',
    3: 'Edge-Ring',
    4: 'Loc',
    5: 'Near-full',
    6: 'Random',
    7: 'Scratch',
    8: 'none'
}

# RESOURCE LOADING 
@st.cache_resource
def load_resources():
    # Load Model
    model = CNN().to(device)
    try:
        # UPDATED FILENAME: cnn_model_aggressive.pth
        model.load_state_dict(torch.load("./data/cnn_model_aggressive.pth", map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    
    # Load Data 
    try:
        df = pd.read_pickle("./data/test_set_aggressive.pkl")
    except:
        st.error("Error loading dataset. Check if 'test_set_aggressive.pkl' exists.")
        return model, None
        
    return model, df

model, df = load_resources()

# REPORT GENERATOR USING LangChain
def generate_report(api_key, defect_type, confidence):
    if not api_key:
        return "Please enter API Key."
    
    context = DEFECT_KNOWLEDGE_BASE.get(defect_type, {})
    
    if defect_type == 'none':
        return "Process Normal. No action needed."

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Updated model name
            google_api_key=api_key,
            temperature=0.2
        )
        
        template = """
        You are a Semiconductor Equipment Engineer.
        
        Task: Write a maintenance ticket based on the data below.
        
        Technical Data:
        Defect: {defect}
        Module: {module}
        Physics: {physics}
        Checklist: {checklist}
        Risk: {risk}
        Confidence: {conf}%
        
        Output Format (Markdown):
        ## Maintenance Ticket: {defect}
        
        **1. Analysis**
        Explain the physics of the defect and possible causes based on the given checklist briefly.
        
        **2. Action Plan**
        Convert the checklist into brief steps
        
        **3. Risk Assessment : **
        State the risk level and what it means in a line
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({
            "defect": defect_type,
            "module": context['Process_Module'],
            "physics": context['Physics'],
            "checklist": str(context['Checklist']),
            "risk": context['Risk_Level'],
            "conf": f"{confidence:.1f}"
        })
        
    except Exception as e:
        return f"Error: {str(e)}"

# DASHBOARD UI
st.title("Automated Fault Detection System")
st.caption(" ResNet style CNN + Generative AI Support")

# Sidebar
api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("### System Knowledge Base")
st.sidebar.json(DEFECT_KNOWLEDGE_BASE)

col1, col2 = st.columns(2)

with col1:
    st.header("1. Detection")
    
    # Button to pick a random wafer from the test set
    if st.button("Scan Random Wafer (Test Set)"):
        if df is not None:
            # Pick one random sample
            sample = df.sample(1).iloc[0]
            st.session_state['sample'] = sample
            st.session_state['report'] = None # Reset report
            
    if 'sample' in st.session_state:
        sample = st.session_state['sample']
        img = sample['waferMap_resized']
        
        # Displaying Image
        fig, ax = plt.subplots(figsize=(3,3))
        # Ensuring we display it correctly (0=purple, 1=yellow)
        ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
        ax.axis('off')
        st.pyplot(fig)
        
        # Running Inference
        img_safe = np.ascontiguousarray(img)
        t = torch.tensor(img_safe, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs, 1)
            
        pred = label_map[idx.item()]
        score = conf.item() * 100
        
        st.session_state['pred'] = pred
        st.session_state['score'] = score
        
        # Displaying Result
        if pred != 'none':
            st.error(f"Detected: {pred}")
        else:
            st.success("Status: Normal")
            
        st.metric("Confidence", f"{score:.1f}%")
        
        # Debug info (optional, remove for production)
        # st.caption(f"Ground Truth Label: {sample['failureType']}")

with col2:
    st.header("2. Resolution")
    
    if 'pred' in st.session_state and st.session_state['pred'] != 'none':
        st.info(f"Retrieving standard procedure for {st.session_state['pred']}...")
        
        if st.button("Generate Maintenance Ticket"):
            with st.spinner("Synthesizing report..."):
                report = generate_report(
                    api_key, 
                    st.session_state['pred'], 
                    st.session_state['score']
                )
                st.session_state['report'] = report
        
        if st.session_state.get('report'):
            st.markdown(st.session_state['report'])