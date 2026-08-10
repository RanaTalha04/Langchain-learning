from dotenv import load_dotenv
from pydantic_class import Person
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate Name, age and city of a person from {place} \n {prompt_structure}",
    input_variables=['place'],
    partial_variables={'prompt_structure': parser.get_format_instructions()}
)

chain = template | model | parser

# prompt = template.invoke({'place': 'Pakistan'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

result = chain.invoke({'place': 'Pakistan'}) 

print(result)