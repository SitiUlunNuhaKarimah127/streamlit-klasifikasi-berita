from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing Summarization
def case_folding(text):
  text = text.lower()
  return text

def cleaning(text):
  text = re.sub(r'[^\w\s.,]', '', text).strip().lower()
  return text

def tokenisasi(text):
  return sent_tokenize(text)

# Preprocessing Klasifikasi
def cleaning_text_klasifikasi(text):
  text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
  return text

def tokenisasi_klasifikasi(text):
  return word_tokenize(text)


def stopwordText_klasifikasi(words):
  corpus = stopwords.words('indonesian')
  return ' '.join([word for word in words if word not in corpus])

def graph_cosine_sim(x, threshold=0.1):
  # TFIDF
  vectorizer = TfidfVectorizer()
  tfidf = vectorizer.fit_transform(x)
  cos_sim = cosine_similarity(tfidf)
  G = nx.Graph()

  # Mengisi nilai similarity antara kalimat ke dalam edges (Garis Penghubung)
  for kalimat in range(len(x)):
    for kalimat_sim in range(kalimat+1, len(x)):
      sim = cos_sim[kalimat][kalimat_sim]
      if sim > threshold:
        G.add_edge(kalimat, kalimat_sim, weight=sim)
  return G

def summarization(x, k = 10, threshold=0.1):
  
  # Memasukkan Nilai Cosine Similarity ke dalam Graph
  G = graph_cosine_sim(x, threshold)

  # Menghitung nilai dari Textrank
  score = nx.pagerank(G)

  # Menyusun Kalimat berdasarkan nilai textrank tertinggi
  score = dict(sorted(score.items(), key=lambda item : item[1], reverse=True))

  summary_sentences = []
  for kalimat, centr in enumerate(score.items()):
    if kalimat < k:
      summary_sentences.append(x[centr[0]])

  return (' '.join(summary_sentences), G)