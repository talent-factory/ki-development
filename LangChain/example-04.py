"""
Mit disem Beispiel wollen wir eine PDF Datei einlesen und
anschliessend einige Fragen zu dessen Inhalt stellen.
"""

import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Get the current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))


def load_pdf(pdf_path):
    """

    Load the PDF document from the specified path.

    :param pdf_path: The path to the PDF file to load.
    :return: The loaded PDF document.

    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def create_vector_store(documents):
    """
    Creates a vector store for the given documents.

    :param documents: A list of documents.
    :type documents: list

    :return: A vector store.
    :rtype: FAISS

    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def create_retrieval_qa_chain(vector_store):
    """
    Creates a retrieval QA chain.

    :param vector_store: A vector store used for retrieval.
    :return: A retrieval QA chain.
    """
    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=vector_store.as_retriever())
    return qa_chain


def main(pdf_path, question):
    """
    :param pdf_path: The path to the PDF document.
    :param question: The question to be answered.
    :return: None

    This method takes in a PDF document path and a question, and performs the following steps:
    1. Loads the PDF document using the `load_pdf` function.
    2. Creates a vector store using the `create_vector_store` function.
    3. Creates a retrieval-QA chain using the `create_retrieval_qa_chain` function.
    4. Invokes the QA chain to answer the question.
    5. Prints the question and the answer.

    Note that this method does not return any value.
    """

    documents = load_pdf(pdf_path)
    vector_store = create_vector_store(documents)
    qa_chain = create_retrieval_qa_chain(vector_store)

    # Beantwortung der Frage
    response = qa_chain.invoke(question)
    print(f"Frage: {question}")
    print(f"Antwort: {response['result']}")


if __name__ == "__main__":
    # Pfad zur PDF-Datei
    pdf_path = os.path.join(current_dir, "../doc", "Script.pdf")

    # Frage zum Dokument
    question = "Was sind die Hauptpunkte des ersten Kapitels?"

    main(pdf_path, question)
