"""
Beispiel für die Verwendung von LangChain zur Beantwortung von Fragen aus PDF-Dokumenten.

Dieses Skript lädt ein PDF-Dokument, erstellt einen Vektorspeicher und ermöglicht
die Beantwortung von Fragen zum Dokumenteninhalt unter Verwendung von OpenAI.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konfiguration
class Config:
    """Konfiguration für die Anwendung."""
    MODEL_TEMPERATURE = 0.3
    CHAIN_TYPE = "stuff"
    DEFAULT_QUESTION = "Was sind die Hauptpunkte des Dokuments?"
    
    # Chunking-Konfiguration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MODEL_NAME = "gpt-3.5-turbo"  # Alternativ: "gpt-4" für grössere Kontextfenster
    MAX_TOKENS = 256  # Maximale Anzahl der Tokens in der Antwort


def find_project_root(current_path: Path) -> Path:
    """
    Rekursiv nach der `pyproject.toml` suchen, um das Projekt-Root-Verzeichnis zu bestimmen.
    
    :param current_path: Startverzeichnis für die Suche
    :return: Pfad zum Projekt-Root-Verzeichnis
    :raises FileNotFoundError: Wenn keine pyproject.toml gefunden wird
    """
    for parent in current_path.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Keine `pyproject.toml`-Datei gefunden. "
        "Stelle sicher, dass du im Projektverzeichnis bist."
    )


def load_environment() -> None:
    """Lädt die Umgebungsvariablen aus der .env-Datei."""
    project_root = find_project_root(Path(__file__))
    dotenv_path = project_root / '.env'
    
    if not dotenv_path.exists():
        logger.warning(f"Keine .env-Datei unter {dotenv_path} gefunden")
        return
    
    load_dotenv(dotenv_path)
    logger.info(f"Umgebungsvariablen aus {dotenv_path} geladen")


def validate_environment() -> None:
    """
    Überprüft, ob alle erforderlichen Umgebungsvariablen gesetzt sind.
    
    :raises ValueError: Wenn erforderliche Umgebungsvariablen fehlen
    """
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Fehlende Umgebungsvariablen: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Lädt ein PDF-Dokument vom angegebenen Pfad.
    
    :param pdf_path: Pfad zur PDF-Datei
    :return: Liste der geladenen Dokumente
    :raises FileNotFoundError: Wenn die PDF-Datei nicht gefunden wird
    :raises Exception: Bei anderen Fehlern beim Laden der PDF
    """
    if not os.path.exists(pdf_path):
        error_msg = f"PDF-Datei nicht gefunden: {pdf_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Lade PDF-Dokument: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Erfolgreich {len(documents)} Seiten geladen")
        return documents
    except Exception as e:
        logger.error(f"Fehler beim Laden der PDF: {str(e)}")
        raise


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Teilt Dokumente in kleinere Chunks auf, um Token-Limits zu vermeiden.
    
    :param documents: Liste der zu teilenden Dokumente
    :return: Liste der geteilten Dokumente
    """
    try:
        logger.info(f"Teile {len(documents)} Dokumente in kleinere Chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Dokumente erfolgreich in {len(split_docs)} Chunks geteilt")
        return split_docs
    except Exception as e:
        logger.error(f"Fehler beim Aufteilen der Dokumente: {str(e)}")
        raise


def create_vector_store(documents: List[Document]) -> VectorStore:
    """
    Erstellt einen Vektorspeicher für die angegebenen Dokumente.
    
    :param documents: Liste der Dokumente
    :return: Vektorspeicher
    :raises Exception: Bei Fehlern beim Erstellen des Vektorspeichers
    """
    try:
        logger.info("Erstelle Vektorspeicher...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("Vektorspeicher erfolgreich erstellt")
        return vector_store
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Vektorspeichers: {str(e)}")
        raise


def create_retrieval_qa_chain(
    vector_store: VectorStore,
    temperature: float = Config.MODEL_TEMPERATURE,
    chain_type: str = Config.CHAIN_TYPE,
    model_name: str = Config.MODEL_NAME
) -> RetrievalQA:
    """
    Erstellt eine Retrieval-QA-Chain für die Beantwortung von Fragen.
    
    :param vector_store: Vektorspeicher für die Dokumentensuche
    :param temperature: Temperatur für die KI-Antwort (0.0 - 1.0)
    :param chain_type: Typ der Chain (z.B. 'stuff', 'map_reduce', 'refine')
    :param model_name: Name des zu verwendenden OpenAI-Modells
    :return: Retrieval-QA-Chain
    :raises ValueError: Wenn der OpenAI-API-Schlüssel fehlt
    """
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = "OPENAI_API_KEY ist nicht gesetzt"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        logger.info(f"Erstelle Retrieval-QA-Chain mit Modell {model_name}...")
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Bei grossen Dokumenten ist 'map_reduce' oder 'refine' oft besser als 'stuff'
        # für Dokumente mit vielen Chunks
        if chain_type == "stuff" and len(vector_store.docstore._dict) > 10:
            logger.info("Viele Dokumente erkannt, empfehle 'map_reduce' statt 'stuff'")
            logger.info("Verwende trotzdem den konfigurierten Chain-Typ")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  # Anzahl der zurückgegebenen Dokumente begrenzen
            return_source_documents=True
        )
        logger.info("Retrieval-QA-Chain erfolgreich erstellt")
        return qa_chain
    except Exception as e:
        logger.error(f"Fehler beim Erstellen der QA-Chain: {str(e)}")
        raise


def process_question(
    qa_chain: RetrievalQA,
    question: str
) -> Dict[str, Any]:
    """
    Verarbeitet eine Frage mit der gegebenen QA-Chain.
    
    :param qa_chain: Die zu verwendende QA-Chain
    :param question: Die zu beantwortende Frage
    :return: Antwort der QA-Chain
    """
    try:
        logger.info(f"Verarbeite Frage: {question}")
        response = qa_chain.invoke({"query": question})
        logger.info("Frage erfolgreich verarbeitet")
        return response
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung der Frage: {str(e)}")
        raise


def main(pdf_path: str, question: str = Config.DEFAULT_QUESTION) -> None:
    """
    Hauptfunktion zum Verarbeiten einer PDF und Beantworten einer Frage.
    
    :param pdf_path: Pfad zur PDF-Datei
    :param question: Zu beantwortende Frage
    """
    try:
        # Umgebung initialisieren
        load_environment()
        validate_environment()
        
        # Dokument laden und verarbeiten
        documents = load_pdf(pdf_path)
        logger.info(f"Gesamtgrösse der Dokumente: {sum(len(doc.page_content) for doc in documents)} Zeichen")
        
        # Dokumente in kleinere Chunks aufteilen, um Token-Limits zu vermeiden
        split_docs = split_documents(documents)
        
        # Vektorspeicher erstellen und Frage beantworten
        vector_store = create_vector_store(split_docs)
        qa_chain = create_retrieval_qa_chain(vector_store)
        
        # Frage beantworten
        response = process_question(qa_chain, question)
        
        # Ausgabe der Ergebnisse
        print(f"\n{'='*80}")
        print(f"Frage: {question}")
        print(f"Antwort: {response['result']}")
        
        # Quellen anzeigen, falls verfügbar
        if 'source_documents' in response and response['source_documents']:
            print("\nQuellen:")
            for i, doc in enumerate(response['source_documents'][:3], 1):
                source = doc.metadata.get('source', 'Unbekannte Quelle')
                page = doc.metadata.get('page', 'N/A')
                print(f"{i}. {source} (Seite {page})")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Ein Fehler ist aufgetreten: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Pfad zur PDF-Datei (relativ zum Skript)
        current_dir = Path(__file__).parent
        pdf_path = current_dir / ".." / "docs" / "index.pdf"
        
        # Frage zum Dokument
        question = "Was sind die Hauptpunkte des Dokuments?"
        
        # Hauptfunktion aufrufen
        main(str(pdf_path), question)
        
    except KeyboardInterrupt:
        print("\nProgramm wurde durch Benutzer unterbrochen")
    except Exception as e:
        logger.error(f"Fehler: {str(e)}")
        print(f"Fehler: {str(e)}")
        exit(1)
