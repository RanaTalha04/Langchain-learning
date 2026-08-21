from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Only write important topic from this directory. {topic}",
    input_variables=['topic']
)
 
loader = DirectoryLoader(
    path="data/Directory",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))

chain = prompt | model | parser

result = chain.invoke({"topic": docs})

print(result)
