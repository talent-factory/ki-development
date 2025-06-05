"""
Hilfsfunktionen für wiederkehrende Projektaufgaben.
"""
import os
from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[str] = None, marker_files=None) -> Path:
    """
    Findet das Projektstammverzeichnis anhand von Markierungsdateien.

    Args:
        start_path: Startverzeichnis für die Suche. Wenn None, wird das Verzeichnis der aktuellen Datei verwendet.
        marker_files: Liste von Dateien/Verzeichnissen, die das Projektstammverzeichnis kennzeichnen.
                    Standard: ['.git', 'pyproject.toml']

    Returns:
        Path: Das gefundene Projektstammverzeichnis.
    """
    if marker_files is None:
        marker_files = ['.git', 'pyproject.toml']

    if start_path is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start_path).resolve()

    current_dir = start_path if start_path.is_dir() else start_path.parent

    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return current_dir  # Fallback to the current directory


def load_environment(env_file: str = '.env', search_from: Optional[str] = None) -> bool:
    """
    Lädt Umgebungsvariablen aus einer .env-Datei im Projektstammverzeichnis.

    Args:
        env_file: Name der zu ladenden Umgebungsdatei.
        search_from: Verzeichnis, von dem aus nach dem Projektstamm gesucht werden soll.

    Returns:
        bool: True, wenn die Umgebungsdatei erfolgreich geladen wurde, sonst False.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv package not installed. Environment variables not loaded.")
        return False

    project_root = find_project_root(search_from)
    env_path = project_root / env_file

    if env_path.exists():
        return load_dotenv(env_path, override=True)
    else:
        # Fallback to loading from current working directory
        return load_dotenv(dotenv_path=env_file, override=True)


# Beispielverwendung:
if __name__ == "__main__":
    # Projektstammverzeichnis finden
    root = find_project_root()
    print(f"Projektstammverzeichnis: {root}")

    # Umgebungsvariablen laden
    load_environment()
    print(f"Umgebungsvariablen geladen von: {os.getenv('ENV_VAR', 'Keine Umgebungsvariablen geladen')}")
