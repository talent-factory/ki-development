#!/usr/bin/env python3
"""
Einfacher Test um zu prüfen ob die Dependencies funktionieren
"""

print("🔍 Teste Dependencies...")

try:
    import numpy as np
    print("✅ NumPy verfügbar")
except ImportError as e:
    print(f"❌ NumPy fehlt: {e}")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Scikit-learn verfügbar")
except ImportError as e:
    print(f"❌ Scikit-learn fehlt: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib verfügbar")
except ImportError as e:
    print(f"❌ Matplotlib fehlt: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers verfügbar")
except ImportError as e:
    print(f"❌ Sentence Transformers fehlt: {e}")

try:
    from openai import OpenAI
    print("✅ OpenAI verfügbar")
except ImportError as e:
    print(f"❌ OpenAI fehlt: {e}")

print("\n🎉 Dependency-Test abgeschlossen!")
