import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Fungsi untuk mengunduh resource NLTK secara senyap
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Panggil fungsi unduh NLTK resource di awal
download_nltk_resources()

# Membaca model yang sudah dilatih dan TF-IDF Vectorizer
logreg_model = joblib.load("model100.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Fungsi untuk mengganti atau menghapus kata-kata tertentu dalam teks
replace_words = {
    "yang": "",
    "nya": "",
    "kok": "",
    "sih": "",
    "ga": "tidak",
    "gak": "tidak",
    "tidakk": "tidak",
    "udah": "sudah",
    "ka": "kak",
    "kakk": "kak",
    "cewe": "cewek",
    "cew": "cewek",
    "cowo": "cowok",
    "cow": "cowok",
}

def replace_and_remove_words(text):
    words = text.split()
    replaced_words = [replace_words.get(word, word) for word in words]
    return " ".join(replaced_words)

# Fungsi untuk membuat word cloud
def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    replaced_text = replace_and_remove_words(" ".join(stemmed_words))  # Replace and remove words
    return replaced_text

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk mengonversi DataFrame ke Excel
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")

# Sidebar untuk navigasi
menu = ["Prediksi Sentimen", "Laporan"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Prediksi Sentimen":
    st.header("Prediksi Sentimen")

    input_text = st.text_input("Masukkan kalimat untuk analisis sentimen:")
    if st.button("Analisis"):
        if input_text.strip() == "":
            st.error("Tolong masukkan sentimen terlebih dahulu.")
        else:
            result = classify_text(input_text)
            st.write("Hasil Analisis Sentimen:", result)

elif choice == "Laporan":
    st.header("Laporan")

    tab1, tab2 = st.tabs(["Prediksi Sentimen", "Analisis Sentimen"])

    with tab1:
        st.header("Unggah file untuk Prediksi Sentimen")
        uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"], key="file_uploader")

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            
            if 'Text' in df.columns:
                X = df['Text'].apply(clean_text)
                X_tfidf = tfidf_vectorizer.transform(X)
                df['Human'] = logreg_model.predict(X_tfidf)
                st.write(df)
                
                st.download_button(
                    label="Unduh file dengan prediksi",
                    data=convert_df_to_excel(df),
                    file_name="prediksi_sentimen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("File harus memiliki kolom 'Text'.")

    with tab2:
        st.header("Unggah file untuk Grafik dan Word Cloud Sentimen")
        uploaded_excel = st.file_uploader("Unggah file Excel", type=["xlsx"], key="file_uploader_analysis")

        if uploaded_excel is not None:
            df_excel = pd.read_excel(uploaded_excel)
            
            if 'Human' in df_excel.columns:
                sentiment_counts = df_excel['Human'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                             labels={'Sentiment': 'Sentimen', 'Count': 'Jumlah'},
                             title='Distribusi Sentimen',
                             text='Count')
                
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                st.plotly_chart(fig)
                
                sentiments = df_excel['Human'].unique()
                for sentiment in sentiments:
                    sentiment_text = " ".join(df_excel[df_excel['Human'] == sentiment]['Text'])
                    sentiment_text_cleaned = clean_text(sentiment_text)
                    create_word_cloud(sentiment_text_cleaned, f'Word Cloud untuk Sentimen {sentiment}')
            else:
                st.error("File harus memiliki kolom 'Human'.")