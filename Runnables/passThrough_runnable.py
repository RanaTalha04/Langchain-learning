from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain_joke': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_chain, parallel_chain)

topic = input("Which topic do you want? ")

result  = chain.invoke({"topic": topic})

print(result['joke'])
print(result['explain_joke'])