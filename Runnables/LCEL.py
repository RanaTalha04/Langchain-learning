# LangChain Expression Language

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

report_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = report_chain | branch_chain

topic = input("Which topic do you want? ")

result  = chain.invoke({"topic": topic})

print(result)