import sys
from pathlib import Path

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Importe nach der Pfadanpassung
from pydantic_ai import Agent  # noqa: E402
from src.utils import load_environment  # noqa: E402

# Load environment variables from .env file in project root
load_environment()


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
