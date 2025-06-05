from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent

def find_project_root():
    """Sucht das Projektstammverzeichnis (enthält .git oder pyproject.toml)."""
    current_dir = Path(__file__).resolve()
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / '.git').exists() or (parent / 'pyproject.toml').exists():
            return parent
    return current_dir  # Fallback zum aktuellen Verzeichnis

# Lade .env Datei aus dem Projektstammverzeichnis
project_root = find_project_root()
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: Versuche, .env im aktuellen Arbeitsverzeichnis zu laden
    load_dotenv()


agent = Agent(
    model="claude-3-7-sonnet-latest",
    system_prompt="Fasse dich kurz und prägnant, antworte auf Deutsch in einem Satz."
)

user_message = "Hallo!"
result = agent.run_sync(user_message)
print(result.output)

while True:
    user_message = input("> ")
    result = agent.run_sync(user_message, message_history=result.all_messages())
    print(result.output)
