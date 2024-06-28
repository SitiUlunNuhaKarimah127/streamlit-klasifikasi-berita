from function import *
import joblib

st.header("Klasifikasi Artikel Berita Berdasarkan Hasil Summarization", divider='rainbow')

text = st.text_area("Masukkan Artikel Berita")

button = st.button("Submit")

if button:
  casefolding = case_folding(text)
  text_clean = cleaning(casefolding)
  tokenizing = tokenisasi(text_clean)
  summary, G = summarization(x=tokenizing, k=10, threshold=0.1)
  
  st.write("**Hasil Summarization:**")
  st.write(summary)
  text_clean_klasifikasi = cleaning_text_klasifikasi(summary)
  tokenizing_klasifikasi = tokenisasi_klasifikasi(text_clean_klasifikasi)
  stopword_removal_klasifikasi = stopwordText_klasifikasi(tokenizing_klasifikasi)
  st.caption("Klasifikasi Berdasarkan Hasil Summarization (SVM)")
  vectorizer = joblib.load("resources/vectorizer_summary.pkl")
  model = joblib.load("resources/Best Model Dengan Peringkasan.pkl")
  
  new_text_matrics = vectorizer.transform([stopword_removal_klasifikasi]).toarray()
  prediction = model.predict(new_text_matrics)
  st.write("Prediction Category : ", prediction[0])
