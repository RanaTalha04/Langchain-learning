from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me name, age, power, city and country of a fictional character from a magic world \n {format_response}",
    input_variables=[],
    partial_variables={'format_response': parser.get_format_instructions()}
)

chain = template | model | parser

# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

result = chain.invoke({})
print(result)