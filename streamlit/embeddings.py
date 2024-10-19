import warnings

import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import logging

# Suppress FutureWarnings from transformers
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="PDF Embeddings & Vector Store",
    page_icon="📄",
    layout="wide"
)


# Load or create embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Extract text from PDF
@st.cache_data
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text


# Split text into chunks
@st.cache_data
def split_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Create embeddings
@st.cache_data
def create_embeddings(chunks, _model):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return _model.encode(chunks)


# Build and save vector store
@st.cache_resource
def build_vector_store(embeddings, chunks):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    vector_store = {
        'index': index,
        'chunks': chunks,
    }
    return vector_store


# Answering user questions
@st.cache_data
def answer_question(question, _vector_store, _model):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        question_embedding = _model.encode([question])
    distances, indices = _vector_store['index'].search(question_embedding, k=3)
    context_chunks = [_vector_store['chunks'][idx] for idx in indices[0]]
    return "\n\n".join(context_chunks)


# Streamlit interface
st.title("PDF Embeddings & Vector Store Tutorial")

st.write("""
Hier zeigen wir, wie man die Funktionsweise von Embeddings und Vector Stores anhand eines Beispiel-PDFs verstehen kann. Laden Sie ein PDF hoch und stellen Sie Fragen zum Inhalt.
""")

uploaded_pdf = st.file_uploader("PDF-Datei hochladen", type="pdf")

if uploaded_pdf is not None:
    # Extract and process PDF
    with st.spinner("Extrahiere Text aus PDF..."):
        text = extract_pdf_text(uploaded_pdf)
        st.write(f"Extrahierter Text:\n\n{text[:500]}...")

    # Split text into chunks
    with st.spinner("Teile Text in Chunks..."):
        chunks = split_text(text)
        st.write(f"Text in {len(chunks)} Chunks unterteilt.")

    # Load the model and create embeddings
    with st.spinner("Lade Modell und erstelle Embeddings..."):
        model = load_model()
        embeddings = create_embeddings(chunks, model)
        st.write("Embeddings erstellt.")

    # Build vector store
    with st.spinner("Erstelle und speichere Vector Store..."):
        vector_store = build_vector_store(np.array(embeddings), chunks)
        st.write("Vector Store erfolgreich erstellt.")

    # User input for questions
    question = st.text_input("Stellen Sie eine Frage zum Dokument:")

    if question:
        with st.spinner("Antwort wird gesucht..."):
            answer = answer_question(question, vector_store, model)
            st.write(f"Antwort:\n\n{answer}")
