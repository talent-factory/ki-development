# Schritt 1: Vorbereitung der Umgebung
# pip install langchain langchain_openai langchain_community openai chromadb tiktoken

# Schritt 2: Importieren der benötigten Module

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Schritt 4: Laden und Vorbereiten der Dokumente
loader = TextLoader("schweiz.txt")
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
query = "Was ist die Hauptstadt der Schweiz?"
result = qa.invoke(query)
print(result['result'])

query = "Welche Sprachen werden in der Schweiz gesprochen?"
result = qa.invoke(query)
print(result['result'])

# Schritt 8: Erweiterung des Systems
new_loader = TextLoader("schweiz_wirtschaft.txt")
new_documents = new_loader.load()
new_texts = text_splitter.split_documents(new_documents)

vectorstore.add_documents(new_texts)

# Schritt 9: Testen des erweiterten Systems
query = "Wofür ist die Schweizer Wirtschaft bekannt?"
result = qa.invoke(query)
print(result['result'])
