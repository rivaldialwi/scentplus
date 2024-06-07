import streamlit as st
import app
import laporan

PAGES = {
    "Aplikasi Analisis Sentimen": app,
    "Laporan Analisis": laporan
}

st.sidebar.title('Navigasi')
selection = st.sidebar.radio("Pilih Halaman", list(PAGES.keys()))

page = PAGES[selection]
page.run()