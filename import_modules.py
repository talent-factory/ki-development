import os
import pkgutil
import re
import subprocess


def is_standard_lib(module_name):
    """Überprüft, ob das Modul Teil der Standardbibliothek ist."""
    return module_name in {name for _, name, _ in pkgutil.iter_modules()}


def find_imported_modules(project_path):
    """Findet alle importierten Module in Python-Dateien."""
    imported_modules = set()
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        match = re.match(r'^\s*(?:import|from) (\S+)', line)
                        if match:
                            module_name = match.group(1).split(".")[0]
                            if not is_standard_lib(module_name):
                                imported_modules.add(module_name)
    return imported_modules


def poetry_add_modules(modules):
    """Fügt jedes Modul mit poetry add hinzu."""
    for module in modules:
        subprocess.run(["poetry", "add", module])


if __name__ == "__main__":
    project_path = "."  # Setzen Sie hier den Pfad zu Ihrem Projektverzeichnis
    imported_modules = find_imported_modules(project_path)
    poetry_add_modules(imported_modules)
    print("Alle gefundenen Module wurden hinzugefügt.")
