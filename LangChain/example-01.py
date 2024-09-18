from langchain.prompts import PromptTemplate

if __name__ == '__main__':
    template = "Schreibe eine kurze Geschichte über einen Helden namens {name}."
    prompt = PromptTemplate(template=template, input_variables=["name"])
    story_prompt = prompt.format(name="Arthur")

    print(story_prompt)
