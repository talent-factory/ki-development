import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Lade Umgebungsvariaben aus der .env-Datei
load_dotenv()

# Überprüfe, ob der OpenAI API-Key gesetzt ist
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Bitte setzen Sie den OPENAI_API_KEY in der .env-Datei oder als Umgebungsvariable.")


def main():
    # Initialisiere das LLM (hier mit einem Standardmodell, das mit der OpenAI-API kompatibel ist)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,  # Kreativität des Modells (0.0 - 1.0)
        max_tokens=500  # Maximale Länge der generierten Antwort
    )

    # Erstelle ein Prompt-Template
    template = """
    Schreibe eine kurze, unterhaltsame Geschichte über einen Helden namens {name}.
    Die Geschichte sollte folgende Elemente enthalten:
    - Einen spannenden Konflikt
    - Eine unerwartete Wendung
    - Ein lehrreiches Ende
    
    Länge: Mindestens 3 Absätze.
    
    Geschichte:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["name"]
    )

    # Erstelle eine LLM-Chain
    story_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True  # Zeigt zusätzliche Informationen an
    )

    # Führe die Chain aus
    hero_name = "Daniel in der Löwengrube"
    print(f"\nGeneriere eine Geschichte über {hero_name}...\n")
    story = story_chain.run(name=hero_name)

    # Zeige das Ergebnis an
    print("\n=== GENERIERTE GESCHICHTE ===\n")
    print(story)
    print("\n" + "=" * 50 + "\n")

    # Optional: Speichere die Geschichte in einer Datei
    with open(f"geschichte_{hero_name.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(f"Geschichte über {hero_name}\n")
        f.write("=" * 30 + "\n\n")
        f.write(story)

    print(f"Die Geschichte wurde in 'geschichte_{hero_name.lower()}.txt' gespeichert.")


if __name__ == "__main__":
    main()
