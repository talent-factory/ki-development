from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent

env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

agent = Agent(
    model="claude-3-7-sonnet-latest",
    system_prompt="Fasse dich kurz und prägnant, antworte auf Deutsch in einem Satz."
)

result = agent.run_sync("Woher komm der Begriff `hello world`?")
print(result.output)
