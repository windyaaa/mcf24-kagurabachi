import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("data.csv")


def home():
    # Vertically align the content
    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h1 style='text-align: center;'>⚜️ IRIS KAGURABACHI</h1>"
        "</div>",
        unsafe_allow_html=True
    )

    st.image('kagurabachi.jpg')
    st.markdown('***')

    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h5>R. Firdaus Dharmawan Akbar</p>"
        "<h5>Dhia Alif Tajriyaani Azhar</p>"
        "<h5>Ridho Pandhu Afrianto</p>"
        "</div>",
        unsafe_allow_html=True
    )


def eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("***")

    st.write('20 rows of the dataset')
    st.write(df.head(20))

# AZHAR'S
def hypothesis_testing():
    st.write("Welcome to the Hypothesis Testing page!")



# Sidebar 
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ("Home", "Visual Analysis", "Hypothesis Testing", "Input Predict","Information Best Model")
)





if selected_page == "Home":
    home()
elif selected_page == "Visual Analysis":
    eda()
elif selected_page == "Hypothesis Testing":
    hypothesis_testing()
elif selected_page == "Input Predict":
    modeling_page()
elif selected_page == "Information Best Model":
    best_model()
