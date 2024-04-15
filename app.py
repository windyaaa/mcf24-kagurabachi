import streamlit as st



def home():
    st.write("IRIS KAGURABACHI")

def eda():
    st.write("Welcome to the Exploratory Data Analysis page!")

# AZHAR'S
def hypothesis_testing():
    st.write("Welcome to the Hypothesis Testing page!")

# DHARMA'S
def modeling():
    st.write("Welcome to the Modeling page!")





# Sidebar 
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ("Home", "Visual Analysis", "Hypothesis Testing", "Modeling")
)





if selected_page == "Home":
    home()
elif selected_page == "Visual Analysis":
    eda()
elif selected_page == "Hypothesis Testing":
    hypothesis_testing()
elif selected_page == "Modeling":
    modeling()
