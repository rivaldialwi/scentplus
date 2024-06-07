import streamlit as st

def main():
    st.sidebar.title("Menu")
    st.sidebar.info("Pilih halaman yang ingin Anda akses.")

    st.title("Aplikasi Analisis Sentimen Scentplus")
    st.write("""
        Selamat datang di aplikasi analisis sentimen Scentplus.
        Silakan pilih menu di sidebar untuk menggunakan aplikasi ini.
    """)

if __name__ == "__main__":
    main()