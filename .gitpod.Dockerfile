# Gitpod Dockerfile für AI Development Kurs
FROM gitpod/workspace-python-3.11

# Installiere zusätzliche System-Pakete
USER root
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    tree \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Wechsle zurück zum gitpod User
USER gitpod

# Installiere Python-Pakete global für bessere Performance
RUN pip install --upgrade pip setuptools wheel

# Installiere häufig verwendete AI/ML Pakete
RUN pip install \
    streamlit \
    openai \
    anthropic \
    python-dotenv \
    langchain \
    langchain-openai \
    langchain-anthropic \
    faiss-cpu \
    chromadb \
    pandas \
    numpy \
    requests \
    jupyter \
    matplotlib \
    seaborn

# Konfiguriere Git (wird von Gitpod überschrieben, aber als Fallback)
RUN git config --global init.defaultBranch main

# Erstelle nützliche Aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -la"' >> ~/.bashrc && \
    echo 'alias streamlit-run="streamlit run"' >> ~/.bashrc && \
    echo 'alias python-version="python --version"' >> ~/.bashrc

# Setze Umgebungsvariablen für bessere Python-Performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Erstelle Arbeitsverzeichnis-Struktur
RUN mkdir -p /home/gitpod/ai-projects
