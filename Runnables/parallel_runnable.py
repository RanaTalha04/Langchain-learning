from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generataion" 
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate a tweet about topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about topic {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

topic = input("Which topic do you want? ")

result  = parallel_chain.invoke({"topic": topic})

print(f"Tweet: {result['tweet']}")
print(f"Linkedin: {result['linkedin']}")