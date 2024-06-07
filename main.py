import streamlit as st
import app
import laporan

PAGES = {
    "Prediksi Sentimen": app,
    "Laporan Analisis Sentimen": laporan
}

st.sidebar.title('Navigasi')
selection = st.sidebar.radio("Pilih Halaman", list(PAGES.keys()))

page = PAGES[selection]
page.run()