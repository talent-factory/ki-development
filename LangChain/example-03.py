from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
result = search.run("What ist Daniel Senften's middle name?")

print(result)
