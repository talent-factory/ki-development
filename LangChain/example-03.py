from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
result = search.run("Wer ist Daniel?")

print(result)
