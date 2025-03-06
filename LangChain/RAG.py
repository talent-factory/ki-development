# Schritt 1: Vorbereitung der Umgebung
# Diese Abhängigkeiten sind jetzt in pyproject.toml definiert und werden über poetry installiert

# Schritt 2: Importieren der benötigten Module
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Umgebungsvariablen laden (für OpenAI API Key)
load_dotenv()

# Schritt 3: Laden und Vorbereiten der Dokumente
current_dir = os.path.dirname(os.path.abspath(__file__))
schweiz_path = os.path.join(current_dir, "schweiz.txt")
loader = TextLoader(schweiz_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=5)
texts = text_splitter.split_documents(documents)

# Schritt 5: Erstellen des Vektorspeichers
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Schritt 6: Erstellen des RAG-Systems
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Schritt 7: Abfragen des RAG-Systems
def run_query(query_text):
    result = qa.invoke(query_text)
    print(f"Frage: {query_text}")
    print(f"Antwort: {result['result']}")
    print("-" * 50)

# Schritt 8: Erweiterung des Systems
def extend_system():
    wirtschaft_path = os.path.join(current_dir, "schweiz_wirtschaft.txt")
    new_loader = TextLoader(wirtschaft_path)
    new_documents = new_loader.load()
    new_texts = text_splitter.split_documents(new_documents)
    vectorstore.add_documents(new_texts)
    print("System mit Wirtschaftsinformationen erweitert.")
    
if __name__ == '__main__':
    # Beispiel-Abfragen
    run_query("Was ist die Hauptstadt der Schweiz?")
    run_query("Welche Sprachen werden in der Schweiz gesprochen?")
    
    # System erweitern
    extend_system()
    
    # Abfrage an das erweiterte System
    run_query("Wofür ist die Schweizer Wirtschaft bekannt?")
