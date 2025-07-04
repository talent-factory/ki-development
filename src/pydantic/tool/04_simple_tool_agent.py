import sys
from pathlib import Path

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Importe nach der Pfadanpassung
from pydantic_ai import Agent  # noqa: E402
from src.utils import load_environment  # noqa: E402

# Lade .env Datei aus dem Projektstammverzeichnis
load_environment()

agent = Agent(
    model="claude-3-7-sonnet-latest",
    system_prompt="Fasse dich kurz und prägnant, antworte auf Deutsch in einem Satz."
)

user_message = "Hallo!"
result = agent.run_sync(user_message)
print(result.output)


# Diese einfache Möglichkeit ein Werkzeug (Tool) zu definieren ist nur eine der Spezialitäten
# von PydanticAI.
#
# Wenn wir die offizielle Seite von Anthropic besuchen, wie die Einbindung von Tools erklärt wird,
# sehen wir, dass es komplexer geht 😉
# https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview

@agent.tool
def get_aktuelle_zeit(self):
    """Gib die aktuelle Zeit im Format '%H:%M:%S' zurück."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


while True:
    user_message = input("> ")
    result = agent.run_sync(user_message, message_history=result.all_messages())
    print(result.output)
