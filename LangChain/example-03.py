from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

def format_search_results(results):
    """Formatiert die Suchergebnisse übersichtlich."""
    if not results:
        return "Keine Suchergebnisse gefunden."
    
    # Wenn es sich um eine Liste von Strings handelt
    if isinstance(results, list):
        formatted_results = []
        for i, item in enumerate(results, 1):
            # Entferne überflüssige Leerzeichen und Zeilenumbrüche
            item = ' '.join(str(item).strip().split())
            formatted_results.append(f"{i}. {item}")
        return "\n\n".join(formatted_results)
    
    # Wenn es sich um ein einzelnes Ergebnis handelt
    return str(results)

def main():
    # Initialisiere die Suchfunktion
    search = SerpAPIWrapper()
    
    # Führe die Suche durch
    query = "Wer ist Daniel?"
    print(f"\n{'='*80}")
    print(f"SUCHERGEBNISSE FÜR: {query}")
    print(f"{'='*80}\n")
    
    try:
        result = search.run(query)
        
        # Formatierte Ausgabe
        formatted_results = format_search_results(result)
        print(formatted_results)
            
    except Exception as e:
        print(f"\nFehler bei der Suche: {str(e)}")
    
    print(f"\n{'='*80}")
    print("Ende der Suchergebnisse")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
