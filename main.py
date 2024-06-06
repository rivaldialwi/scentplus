import streamlit as st
import importlib

st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu", ["Prediksi Sentimen", "Laporan"])

if menu == "Prediksi Sentimen":
    st.title("Aplikasi Analisis Sentimen Scentplus - Prediksi Sentimen")
    module = importlib.import_module('app')
    module.run()

elif menu == "Laporan":
    st.title("Aplikasi Analisis Sentimen Scentplus - Laporan")
    module = importlib.import_module('laporan')
    module.run()