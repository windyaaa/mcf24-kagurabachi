import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import shap
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

# DHARMA'S
def preprocess_dataframe(df):
    df['Jenis Kelamin'] = df['Jenis Kelamin'].replace({'F': 0, 'M': 1})
    df['Jenis Kelamin'] = df['Jenis Kelamin'].astype(int)

    bins = [0, 18.5, 25, 30, float('inf')]
    labels = ['Kurus', 'Normal', 'Kegemukan', 'Obesitas']
    df['IMT_Category'] = pd.cut(df['IMT (kg/m2)'], bins=bins, labels=labels, right=False)

    usia_global_bins = [0, 13, 20, 40, 60, float('inf')]
    usia_global_labels = [0, 1, 2, 3, 4]
    df['Usia_Category'] = pd.cut(df['Usia'], bins=usia_global_bins, labels=usia_global_labels, right=False)
    df['Usia_Category'] = df['Usia_Category'].astype(int)

    columns_to_encode = ['IMT_Category']
    df_encoded = pd.get_dummies(df, columns=columns_to_encode)

    return df_encoded

def engineering(X, y, test_size=0.25, random_state=42):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Handling outliers using LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(X_train.iloc[:, 1:11])
    outlier_labels_train = lof.fit_predict(X_train.iloc[:, 1:11])
    outliers_indices_train = np.where(outlier_labels_train == -1)[0]
    for feature_index in range(X_train.iloc[:, 1:11].shape[1]):  
        median_value = np.median(X_train.iloc[:, 1:11].iloc[:, feature_index])  
        X_train.iloc[outliers_indices_train, 1:11].iloc[:, feature_index] = median_value 
    
    # Scaling features using PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled.iloc[:, 1:11] = pt.fit_transform(X_train.iloc[:, 1:11])
    X_test_scaled.iloc[:, 1:11] = pt.transform(X_test.iloc[:, 1:11])
    
    # Resampling using SMOTEN
    oversampler = SMOTEN(random_state=random_state)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)
    
    # Convert IMT_Category columns to int
    X_train_resampled['IMT_Category_Kurus'] = X_train_resampled['IMT_Category_Kurus'].astype('int')
    X_train_resampled['IMT_Category_Normal'] = X_train_resampled['IMT_Category_Normal'].astype('int')
    X_train_resampled['IMT_Category_Kegemukan'] = X_train_resampled['IMT_Category_Kegemukan'].astype('int')
    X_train_resampled['IMT_Category_Obesitas'] = X_train_resampled['IMT_Category_Obesitas'].astype('int')
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

def model_fit(X_train_resampled, y_train_resampled):
    param_dist = {
        'colsample_bytree': 0.5258177389372565,
        'gamma': 0.18242481428361046,
        'learning_rate': 0.21908843870992648,
        'max_depth': 9,
        'n_estimators': 105,
        'reg_alpha': 0.39785559904574164,
        'reg_lambda': 0.9694704332753689,
        'subsample': 0.9327535629469901,
        'random_state': 42
    }

    xgb_model = xgb.XGBClassifier(**param_dist)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    return xgb_model

def model_predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def modeling_page():
    st.header("Welcome to the Modeling page!")
    tab1, tab2 = st.tabs(["Information Best Model", "Input Predict"])
    with tab1:
        raw_data = pd.read_csv('data.csv').drop(columns=['Responden'],axis=True)

        df = raw_data.copy().drop(columns=['Tempat lahir'],axis=True)

        threshold = 200
        df['CT_Category'] = np.where(df['Cholesterol Total (mg/dL)'] < threshold, 0, 1)
        df['CT_Category'] = df['CT_Category'].astype(int)

        df_encoded = preprocess_dataframe(df)

        df_encoded.drop(columns=['Cholesterol Total (mg/dL)', 'Usia'], axis=1, inplace=True)

        X_train_resampled, X_test_scaled, y_train_resampled, y_test = engineering(df_encoded.drop(columns=['CT_Category']), df_encoded['CT_Category'])

        model = model_fit(X_train_resampled, y_train_resampled)

        y_pred_test = model_predict(model, X_test_scaled)

        report_xgb_str = "XGBoost:\n" + classification_report(y_test, y_pred_test)
        
        st.write('**Detailed information performace best model (XGBoost) with 25% test data**')

        st.header('Confusion Matrix Best Model')
        col3, col4,col5 = st.columns([0.2,0.6,0.2])
        with col4:
            cm = confusion_matrix(y_test, y_pred_test)

            tick_labels = ['Normal', 'Tinggi']

            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=tick_labels, yticklabels=tick_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot()

        st.header('Metrics Evaluation Best Model')
        st.code(f"{report_xgb_str}", language='python')


        st.header('Interpretation/Explainable Best Model With SHAP Values')
        col1, col2 = st.columns([0.5,0.5])
        with col1:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train_resampled.columns)
            plt.title("Summary Plot")
            st.pyplot()

        with col2:
            shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train_resampled.columns, plot_type='bar')
            plt.title("Summary Plot (Bar)")
            st.pyplot()



    with tab2:
        st.header('Input Predict')
        # User Input for Prediction
        jenis_kelamin = st.radio("Jenis Kelamin:", ('Perempuan', 'Laki-laki'))

        jenis_kelamin = 'F' if jenis_kelamin == 'Perempuan' else 'M'


        usia = st.number_input("Usia:", min_value=0, step=1)
        
        tekanan_darah_s = st.number_input("Tekanan darah (S):", min_value=0.0)
        
        tekanan_darah_d = st.number_input("Tekanan darah (D):", min_value=0.0)
        
        tinggi_badan = st.number_input("Tinggi badan (cm):", min_value=0.0)
        
        berat_badan = st.number_input("Berat badan (kg):", min_value=0.0)
        
        imt = st.number_input("IMT (kg/m2):", min_value=0.0)
        
        lingkar_perut = st.number_input("Lingkar perut (cm):", min_value=0.0)
        
        glukosa_puasa = st.number_input("Glukosa Puasa (mg/dL):", min_value=0.0)
        
        trigliserida = st.number_input("Trigliserida (mg/dL):", min_value=0.0)
        
        fat = st.number_input("Fat", min_value=0.0)
        
        visceral_fat = st.number_input("Visceral Fat", min_value=0.0)
        
        masa_kerja = st.number_input("Masa Kerja:", min_value=0.0)

        input_user = {
                'Jenis Kelamin': [jenis_kelamin],
                'Usia': [usia],
                'Tekanan darah  (S)': [tekanan_darah_s],
                'Tekanan darah  (D)': [tekanan_darah_d],
                'Tinggi badan (cm)': [tinggi_badan],
                'Berat badan (kg)': [berat_badan],
                'IMT (kg/m2)': [imt],
                'Lingkar perut (cm)': [lingkar_perut],
                'Glukosa Puasa (mg/dL)': [glukosa_puasa],
                'Trigliserida (mg/dL)': [trigliserida],
                'Fat': [fat],
                'Visceral Fat': [visceral_fat],
                'Masa Kerja': [masa_kerja]
            }
        df_input = pd.DataFrame(input_user)

        preprocessed_df_input = preprocess_dataframe(df_input)
        preprocessed_df_input.drop(columns=['Usia'], axis=1, inplace=True)
        # st.write(preprocessed_df_input)

        y_pred_input = model_predict(model, preprocessed_df_input)

        if st.button('Submit'):
            st.write('---')
            st.header('Prediction Result')

            prediction_label = 'Tinggi' if y_pred_input[0] == 1 else 'Normal'

            if prediction_label == 'Tinggi':
                st.markdown(f'Cholesterol Total Category: ')
                st.error(prediction_label)
            else:
                st.markdown(f'Cholesterol Total Category: ')
                st.success(prediction_label)





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
    modeling_page()
