import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


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
        "<h5>R. Firdaus Dharmawan Akbar</h5>"
        "<h5>Dhia Alif Tajriyaani Azhar</h5>"
        "<h5>Ridho Pandhu Afrianto</h5>"
        "</div>",
        unsafe_allow_html=True
    )


def eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("***")

    # st.text("")
    st.markdown(
        "<h5>The head of the data</h5>",  
        unsafe_allow_html=True
    )
    st.write(df.head(20))
    # Count respondents with high and normal cholesterol levels
    high_cholesterol_count = df[df['Cholesterol Total (mg/dL)'] >= 200].shape[0]
    normal_cholesterol_count = df[df['Cholesterol Total (mg/dL)'] < 200].shape[0]

    with st.expander("❗ Pre-indicator for imbalanced data"):
        # st.write(f"Number of respondents with normal cholesterol (< 200 mg/dL): {normal_cholesterol_count}")
        # st.write(f"Number of respondents with high cholesterol (>= 200 mg/dL): {high_cholesterol_count}")

        # Create data for pie chart
        labels = ['Normal Cholesterol', 'High Cholesterol']
        counts = [normal_cholesterol_count, high_cholesterol_count]

        # Create a pie chart using Plotly with reduced size
        fig = px.pie(values=counts, names=labels, title="Cholesterol Distribution", 
                    labels={'label': 'Cholesterol Level', 'value': 'Count'})

        # Set width and height of the figure
        fig.update_layout(width=800, height=800)

        # Display the pie chart
        st.plotly_chart(fig)  


    with st.expander("💡 Variables explanation"):
        st.write("""
        1. Responden: Merupakan identitas unik bagi setiap pegawai yang mengikuti survey.

2. Jenis Kelamin: Kategorikal, menunjukkan jenis kelamin pegawai, bisa berupa "Laki-laki" atau "Perempuan".

3. Usia: Numerik, menunjukkan usia pegawai dalam satuan tahun.

4. Tekanan Darah (S): Numerik, menunjukkan nilai tekanan darah sistolik pegawai dalam satuan mmHg. Tekanan darah sistolik adalah tekanan darah saat jantung berkontraksi.

5. Tekanan Darah (D): Numerik, menunjukkan nilai tekanan darah diastolik pegawai dalam satuan mmHg. Tekanan darah diastolik adalah tekanan darah saat jantung berelaksasi.

6. Tinggi Badan (cm): Numerik, menunjukkan tinggi badan pegawai dalam satuan sentimeter.

7. Berat Badan (kg): Numerik, menunjukkan berat badan pegawai dalam satuan kilogram.

8. IMT (kg/m2): Numerik, menunjukkan Indeks Massa Tubuh (IMT) pegawai, dihitung dengan rumus: berat badan (kg) / [tinggi badan (m)]^2. IMT digunakan untuk mengklasifikasikan status berat badan (kurus, normal, overweight, obesitas).

9. Lingkar Perut (cm): Numerik, menunjukkan lingkar perut pegawai dalam satuan sentimeter. Lingkar perut digunakan sebagai indikator risiko kesehatan metabolik.

10. Glukosa Puasa (mg/dL): Numerik, menunjukkan kadar glukosa darah pegawai saat puasa (tidak makan minimal 8 jam) dalam satuan miligram per desiliter (mg/dL). Kadar glukosa darah yang tinggi dapat mengindikasikan diabetes.

11. Kolesterol Total (mg/dL): Numerik, menunjukkan kadar kolesterol total dalam darah pegawai dalam satuan miligram per desiliter (mg/dL). Kolesterol total terdiri dari kolesterol LDL ("jahat") dan HDL ("baik").

12. Trigliserida (mg/dL): Numerik, menunjukkan kadar trigliserida dalam darah pegawai dalam satuan miligram per desiliter (mg/dL). Trigliserida adalah jenis lemak yang disimpan dalam tubuh. Kadar trigliserida yang tinggi dapat meningkatkan risiko penyakit jantung.

13. Fat: Numerik, menunjukkan persentase lemak tubuh pegawai.

14. Visceral Fat: Numerik, menunjukkan persentase lemak visceral (lemak di sekitar organ perut) pegawai. 

15. Masa Kerja: Numerik, menunjukkan lama waktu pegawai bekerja di perusahaan dalam satuan tahun.

16. Tempat Lahir: Kategorikal, menunjukkan kota tempat lahir pegawai.
            """)
    ##############################

    ##### DESC STATS #####
    st.text("")
    st.markdown(
        "<br>"
        "<h5>Descriptive Statistics</h5>",  
        unsafe_allow_html=True
    )
    st.write(df.describe())
    

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            * Dari data tersebut, terlihat bahwa mayoritas responden berusia relatif muda, dengan beragam variabilitas dalam parameter kesehatan seperti tekanan darah, tinggi badan, berat badan, dan kadar gula darah. 
            * Dapat terlihat ada variasi dalam kadar kolesterol dan trigliserida. 
            * Meskipun demikian, untuk beberapa variabel seperti tinggi badan dan kadar glukosa puasa, terdapat konsistensi yang mencolok dalam nilai median.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Statistika deskriptif diperlukan untuk dataset ini agar kita dapat memahami dan menggambarkan secara ringkas karakteristik-karakteristik utama dari data kesehatan pegawai yang disurvei. 
            * Informasi ini berguna untuk memberikan pemahaman awal tentang profil kesehatan populasi pegawai tersebut, yang dapat menjadi dasar untuk analisis lebih lanjut,seperti pengambilan keputusan ataupun pemodelan.
                """)
    ########################################


    ##### DISTRIBUTIONS #####

    st.markdown(
        "<br>"
        "<h5>Distributions</h5>",  
        unsafe_allow_html=True
    )


    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                                and col not in ['Responden', 'Tempat Lahir']]

    num_plots = len(numerical_columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Define a list of colors
    colors = sns.color_palette("husl", num_plots)

    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_plots:
                with cols[j]:
                    col = numerical_columns[idx]
                    st.write(col)
                    # Adjust the plotting parameters
                    if col == "Usia":  # Adjust parameters for 'Usia' plot
                        sns.histplot(data=df, x=col, kde=True, color=colors[idx])
                    else:  # For other plots
                        sns.histplot(data=df, x=col, kde=True, color=colors[idx])
                    plt.xlabel('')
                    st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)

    #################################

    ##### CORR MATRIX #####
    st.markdown(
        "<br>"
        "<h5>Correlation Matrix</h5>",  
        unsafe_allow_html=True
    )
    correlation_matrix = df[numerical_columns].corr()

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    #################################



    ##### Tekanan Darah #####
    st.text("")

    st.markdown(
        "<br>"
        "<h5>Blood Pressure by Age</h5>",  
        unsafe_allow_html=True
    )

    # Calculate Mean Arterial Pressure (MAP)
    df['Mean Arterial Pressure'] = (2 * df['Tekanan darah  (D)'] + df['Tekanan darah  (S)']) / 3

    # Filter respondents based on 'Cholesterol Total (mg/dL)'
    high_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] >= 200]
    low_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] < 200]

    custom_palette = {"M": "blue", "F": "red"}

    # Column 1 for respondents with high cholesterol
    col1, col2 = st.columns(2)
    with col1:
        st.write('High Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=high_cholesterol_df, x="Usia", y="Mean Arterial Pressure", hue="Jenis Kelamin", palette=custom_palette)
        # plt.title("Mean Arterial Pressure (High Cholesterol)")
        plt.xlabel("Age")
        plt.ylabel("Mean Arterial Pressure")
        plt.legend(title="Gender")
        st.pyplot()

    # Column 2 for respondents with low cholesterol
    with col2:
        st.write('Normal Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=low_cholesterol_df, x="Usia", y="Mean Arterial Pressure", hue="Jenis Kelamin", palette=custom_palette)
        # plt.title("Mean Arterial Pressure (Low Cholesterol)")
        plt.xlabel("Age")
        plt.ylabel("Mean Arterial Pressure")
        plt.legend(title="Gender")
        st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)

    ###############################################



    ##### FASTING GLUCOSE #####

    st.text("")

    st.markdown(
        "<br>"
        "<h5>Fasting Glucose by Age</h5>",  
        unsafe_allow_html=True
    )


    # Filter respondents based on 'Cholesterol Total (mg/dL)'
    high_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] >= 200]
    low_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] < 200]

    # Define custom color palettes
    # colors = sns.color_palette("Set2")
    custom_palette = {"M": "orange", "F": "green"}

    # Column 1 for respondents with high cholesterol
    col1, col2 = st.columns(2)
    with col1:
        st.write('High Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=high_cholesterol_df, x="Usia", y="Glukosa Puasa (mg/dL)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Glukosa Puasa")
        plt.legend(title="Gender")
        st.pyplot()

    # Column 2 for respondents with low cholesterol
    with col2:
        st.write('Normal Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=low_cholesterol_df, x="Usia", y="Glukosa Puasa (mg/dL)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Glukosa Puasa")
        plt.legend(title="Gender")
        st.pyplot()


    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    #################################



        ##### FASTING GLUCOSE #####

    st.text("")

    st.markdown(
        "<br>"
        "<h5>Body Mass Index by Age</h5>",  
        unsafe_allow_html=True
    )


    # Filter respondents based on 'Cholesterol Total (mg/dL)'
    high_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] >= 200]
    low_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] < 200]

    # Define custom color palettes
    # colors = sns.color_palette("Set1")
    custom_palette = {"M": "blue", "F": "orange"}

    # Column 1 for respondents with high cholesterol
    col1, col2 = st.columns(2)
    with col1:
        st.write('High Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=high_cholesterol_df, x="Usia", y="IMT (kg/m2)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Glukosa Puasa")
        plt.legend(title="Gender")
        st.pyplot()

    # Column 2 for respondents with low cholesterol
    with col2:
        st.write('Normal Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=low_cholesterol_df, x="Usia", y="IMT (kg/m2)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Glukosa Puasa")
        plt.legend(title="Gender")
        st.pyplot()


    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    #################################

        ##### FASTING GLUCOSE #####

    st.text("")

    st.markdown(
        "<br>"
        "<h5>Triglyceride by Age</h5>",  
        unsafe_allow_html=True
    )


    # Filter respondents based on 'Cholesterol Total (mg/dL)'
    high_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] >= 200]
    low_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] < 200]

    # Define custom color palettes
    # colors = sns.color_palette("husl")
    custom_palette = {"M": "red", "F": "orange"}
    

    # Column 1 for respondents with high cholesterol
    col1, col2 = st.columns(2)
    with col1:
        st.write('High Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=high_cholesterol_df, x="Usia", y="Trigliserida (mg/dL)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Trigliserida")
        plt.legend(title="Gender")
        st.pyplot()

    # Column 2 for respondents with low cholesterol
    with col2:
        st.write('Normal Cholesterol')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=low_cholesterol_df, x="Usia", y="Trigliserida (mg/dL)", hue="Jenis Kelamin", palette=custom_palette)
        plt.xlabel("Age")
        plt.ylabel("Trigliserida")
        plt.legend(title="Gender")
        st.pyplot()


    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    #################################


    ##### MEAN COMPARISON VALUE #####

    st.markdown(
        "<br>"
        "<h5>Mean Value Comparison</h5>",  
        unsafe_allow_html=True
    )

    # Define the features for comparison
    features = ['Tekanan darah  (D)', 'Tekanan darah  (S)', 'Glukosa Puasa (mg/dL)', 'IMT (kg/m2)', 
                'Trigliserida (mg/dL)', 'Fat', 'Visceral Fat', 'Masa Kerja']

    # Filter respondents based on 'Cholesterol Total (mg/dL)'
    high_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] >= 200]
    low_cholesterol_df = df[df['Cholesterol Total (mg/dL)'] < 200]

    # Calculate mean values for each feature
    high_mean_values = high_cholesterol_df[features].mean()
    low_mean_values = low_cholesterol_df[features].mean()

    # Combine mean values into a DataFrame
    mean_values_df = pd.DataFrame({'Feature': features,
                                   'High Cholesterol': high_mean_values,
                                   'Normal Cholesterol': low_mean_values})

    # Melt the DataFrame to plot with Seaborn
    mean_values_df_melted = mean_values_df.melt(id_vars='Feature', var_name='Cholesterol Level', value_name='Mean Value')

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=mean_values_df_melted, x='Feature', y='Mean Value', hue='Cholesterol Level', marker='o')
    plt.xticks(rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Mean Value')
    plt.title('Mean Value Comparison between Normal and High Cholesterol')
    plt.legend(title='Cholesterol Level')

    # Add grid
    plt.grid(True)

    st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            ===== TO BE DETERMINED =====
                """)

    #####################################

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
