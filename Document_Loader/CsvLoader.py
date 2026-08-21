from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Explain what type of data is in this file. {file}",
    input_variables=['file']
)
 
loader = CSVLoader(file_path = "data/heart.csv")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"file": docs})

print(result)
