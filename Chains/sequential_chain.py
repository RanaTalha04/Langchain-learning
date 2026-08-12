from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generataion" 
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 poin summary from a followig text \n {text}",
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'RAG'})

print(result, end="\n\n")


chain.get_graph().print_ascii()