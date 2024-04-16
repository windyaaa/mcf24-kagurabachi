import streamlit as st
import pandas as pd
import seaborn as sns
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


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
        fig.update_layout(width=400, height=400)

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
            Distribusi data secara visual tidak ada yang normal, bahkan untuk variabel kolesterol total sekalipun.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Distribusi data ini penting karena memberikan informasi tentang pola-pola umum, termasuk apakah data cenderung terkumpul di sekitar nilai-nilai tertentu atau apakah ada outlier yang signifikan. 
            * Informasi ini membantu kita dalam mengidentifikasi tren, anomali, dan karakteristik khusus dari populasi yang disurvei. 
            * Dengan pemahaman yang lebih baik tentang distribusi variabel, kita dapat membuat keputusan yang lebih tepat dan merancang strategi intervensi yang lebih efektif dalam konteks kesehatan pegawai.
                """)

    #################################

    ##### SCATTER PLOT - LINEARITY #####
    st.markdown(
        "<br>"
        "<h5>Scatter Plot</h5>",  
        unsafe_allow_html=True
    )

    # Select numerical columns (excluding 'Responden', 'Usia', 'Tempat Lahir', and 'Jenis Kelamin')
    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                        and col not in ['Responden', 'Usia', 'Tempat Lahir', 'Jenis Kelamin']]

    num_plots = len(numerical_columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    colors = sns.color_palette("husl", num_plots)

    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_plots:
                with cols[j]:
                    col = numerical_columns[idx]
                    st.write(f" {col} vs Cholesterol Total")
                    sns.scatterplot(data=df, x=col, y='Cholesterol Total (mg/dL)', color=colors[idx])
                    plt.xlabel(col)
                    plt.ylabel('Cholesterol Total (mg/dL)')
                    plt.grid(True)
                    st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
             Dapat dilihat bahwa variabel-variabel ini tidak bersifat linear, kecuali variabel y terhadap variabel itu sendiri, yaitu kolesterol total
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Scatter plot digunakan untuk secara jelas memberikan visualisasi yang langsung tentang hubungan antara dua variabel 
                     dan memungkinkan untuk dengan cepat menentukan apakah hubungan tersebut cenderung linier atau tidak,
                    serta mendeteksi adanya outlier yang dapat mempengaruhi kecenderungan linieritas antara kedua variabel tersebut.
                """)
    #####################################

    ##### CORR MATRIX #####
    st.markdown(
        "<br>"
        "<h5>Correlation Matrix</h5>",  
        unsafe_allow_html=True
    )

    # Create a label encoder object
    label_encoder = LabelEncoder()

    # Encode the 'Jenis Kelamin' column
    df['Jenis Kelamin Encoded'] = label_encoder.fit_transform(df['Jenis Kelamin'])
    df_encode = df.drop(columns=['Responden','Jenis Kelamin','Tempat lahir'])

    # Generate Spearman correlation matrix
    correlation_matrix_spearman = df_encode.corr(method='spearman')

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f")
    plt.tight_layout()
    st.pyplot()


    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            Correlation matrix terbukti memiliki banyak pasangan variabel yang saling berhubungan positif. Pasangan variabel IMT dengan berat badan menunjukkan hubungan yang kuat dengan nilai 0.87. sedangkan pasangan variabel jenis kelamin encoded dengan variabel fat menunjukkan adanya hubungan negatif yang cukup kuat diantaranya, dengan nilai -0.30.

                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Matriks korelasi memberikan informasi tentang sejauh mana dua variabel bergerak bersama-sama, apakah hubungan antara mereka positif (seiring bertambahnya nilai satu variabel, nilai variabel lainnya juga meningkat), negatif (salah satu nilai meningkat, yang lainnya menurun), atau tidak berkorelasi sama sekali. 
            * Spearman dipilih menjadi metode untuk kalkulasi korelasi dikarenakan tidak ada variabel yang linier (terbukti pada scatter plot).
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
            Plot ini membuktikan bahwa Perempuan cenderung memiliki rata-rata tekanan darah lebih rendah daripada laki-laki pada kedua tingkat kolesterol normal dan tinggi. tingkat tertinggi tekanan darah terdapat pada laki-laki di usia 25-27, sedangkan tingkat terendah pada kolesterol normal terdapat pada perempuan di usia 20-22.

                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
           Visualisasi ini dipilih karena sesuai penelitian Madsen et. al. (2017), faktor tekanan darah memiliki hubungan sebab akibat dengan kolesterol total. 
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
            Terbukti bahwa pada kedua tingkat kolesterol, perempuan dan laki-laki memiliki nilai glukosa puasa yang relatif mirip. namun pada kolesterol tinggi, laki-laki pada usia 37 tahun memiliki nilai kolesterol yang paling tinggi. lain halnya dengan kolesterol normal, perempuan pada usia 32 tahun memiliki nilai kolesterol tertinggi.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            Visualisasi ini dipilih karena sesuai penelitian Madsen et. al. (2017), faktor glukosa pada saat puasa memiliki hubungan dengan kolesterol total. 
                """)
    #################################



        ##### BMI #####

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
Plot tersebut  menunjukkan bahwa nilai BMI untuk tingkat kolesterol tinggi cenderung meningkat seiring bertambahnya usia, terutama pada wanita. sedangkan pada tingkat kolesterol normal, nilai BMI pada laki-laki cenderung meningkat seiring bertambahnya usia.

                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
             Visualisasi ini dipilih karena sesuai penelitian Madsen et. al. (2017), faktor glukosa pada saat puasa memiliki hubungan dengan kolesterol total. 
                """)
    #################################

        ##### TRIGLYCERIDE #####

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
Plot ini menunjukkan bahwa high cholesterol memiliki nilai triglyceride yang berbeda untuk kedua gender mengalami fluktuasi naik-turun yang signifikan. sedangkan pada normal cholesterol, untuk kedua gender memiliki tingkat triglyceride yang konstan dari usia 20-27 tahun, namun mulai mengalami perubahan pada usia 30 tahun, dengan tingkat triglyseride tertinggi oleh laki-laki dengan usia 37 tahun.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            Visualisasi ini dipilih karena sesuai penelitian Madsen et. al. (2017), faktor glukosa pada saat puasa memiliki hubungan erat dengan kolesterol total. 
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
            Dari plot ini, terdapat kesamaan nilai mean value pada variabel-variabel yang diuji kecuali pada variabel variabel trigliserida yang menunjukkan perbedaan nilai mean value yang paling besar, dengan tingkat kolesterol normal dengan nilai 100 dan tingkat kolesterol tinggi dengan nilai 140.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Plot ini berguna sebagai acuan kita untuk mengetahui sebenarnya berapa nilai rata-rata dari setiap variabel penting pada orang dengan kolesterol normal.
            * Dengan membandingkannya pada nilai-nilai variabel penderita kolesterol tinggi, kita dapat mengetahui seberapa jauh perbedaan tersebut.
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
