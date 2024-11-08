import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Laden der Umgebungsvariablen aus der .env-Datei
load_dotenv()

# key = os.environ.get("OPENAI_API_KEY")
# print(key)


# noinspection PyShadowingNames
class NewChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs):
        formatted_prompt = self.prompt.format(**inputs)
        return self.llm.invoke(formatted_prompt)


if __name__ == '__main__':
    template = "Schreibe eine kurze Geschichte über einen Helden namens {name}."
    prompt = PromptTemplate(template=template, input_variables=["name"])
    name = "Jürg"
    story_prompt = prompt.format(name=name)

    llm = OpenAI()  # get the OpenAI API Key from environment variable
    chain = NewChain(llm=llm, prompt=prompt)
    story = chain.invoke({"name": name})

    print(story)
