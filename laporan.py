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
from sklearn.metrics import accuracy_score, precision_score, recall_score

def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_resources()

logreg_model = joblib.load("model100.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

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

def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]
    replaced_text = replace_and_remove_words(" ".join(stemmed_words))
    return replaced_text

def classify_text(input_text):
    cleaned_text = clean_text(input_text)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

def run():
    st.title("Aplikasi Analisis Sentimen Scentplus")

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