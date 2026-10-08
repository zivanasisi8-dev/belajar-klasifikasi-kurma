import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Belajar Klasifikasi Kurma",
	page_icon = ":date:"
)

model = joblib.load("model_klasifikasi_kurma.joblib")

st.title(":date: Belajar Klasifikasi Kurma")
st.markdown("Aplikasi machine learning classification untuk memprediksi stage kurma")

ukuran_cm = st.slider("Ukuran (cm)", 1.5, 6.5, 4.0)
berat_g = st.slider("Berat (g)", 3.0, 30.0, 15.0)
kekerasan_N = st.slider("Kekerasan (N)", 0.0, 20.0, 10.0)
kadar_gula_brix = st.slider("Kadar Gula (Brix)", 10.0, 80.0, 30.0)
kadar_air_pct = st.slider("Kadar Air (%)", 10.0, 80.0, 40.0)
varietas = st.pills("Varietas", ["Ajwa","Barhi","Deglet Noor","Medjool","Zahidi"], default="Ajwa")
tingkat_warna = st.pills("Tingkat Warna", ["Hitam","Cokelat Muda","Kuning","Cokelat Tua"], default="Kuning")
grade_cacat = st.pills("Grade Cacat", ["Ringan","Berat","Tidak Ada","Sedang"], default="Tidak Ada")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[ukuran_cm,berat_g,kekerasan_N,kadar_gula_brix,kadar_air_pct,varietas,tingkat_warna,grade_cacat]], columns=["ukuran_cm","berat_g","kekerasan_N","kadar_gula_brix","kadar_air_pct","varietas","tingkat_warna","grade_cacat"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :date: oleh **Safira Nur Azzahra**")
