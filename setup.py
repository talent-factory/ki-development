from setuptools import setup, find_packages

setup(
    name="ai_development",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # Fügen Sie hier Ihre Abhängigkeiten ein
        "python-dotenv",
        "pydantic-ai",
    ],
)
