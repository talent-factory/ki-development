from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent

# Lade .env Datei aus dem Projektstammverzeichnis
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)


agent = Agent(
    model="claude-3-7-sonnet-latest",
)

result = agent.run_sync("Wer ist der Vorsitzende der Bundesversammlung?")
print(result.output)
