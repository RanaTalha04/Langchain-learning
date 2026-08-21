from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write the summary for given poem. {poem}",
    input_variables=['poem']
)
 
loader = TextLoader("data/cricket.txt", encoding="utf-8")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})

print(result)
