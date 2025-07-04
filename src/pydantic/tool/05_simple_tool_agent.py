import sys
import os
from pathlib import Path
from datetime import datetime
import googleapiclient.discovery

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
    system_prompt="Du bist ein YouTube Experte. Du suchst bestimmte Videos auf YouTube. Nutze die Tools, um die Videos zu finden."
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
    return datetime.now().strftime("%H:%M:%S")


@agent.tool
def youtube_suche(self, suchbegriff):
    """Suche auf YouTube nach Videos mit dem Suchbegriff.
    
    Ref: https://developers.google.com/youtube/v3/docs/search/list?hl=de
    
    Args:
        suchbegriff: Der Suchbegriff für die YouTube-Suche.
    
    Returns:
        list: Liste von Videos mit Titel und URL.
    """
    api_service_name = "youtube"
    api_version = "v3"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=os.environ.get("YOUTUBE_API_KEY")
    )

    request = youtube.search().list(
        part="id,snippet", q=suchbegriff, maxResults=3, type="video"
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video = {
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
        }
        videos.append(video)

    return videos


while True:
    user_message = input("> ")
    result = agent.run_sync(user_message, message_history=result.all_messages())
    print(result.output)
