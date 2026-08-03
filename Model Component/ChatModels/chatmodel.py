from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model = 'llama3.2:latest',
    temperature = 0.2,
    num_predict = 256
)

messages = [
    SystemMessage(content="You are an AI assistant that speaks exclusively like a 1920s pirate."),
    HumanMessage(content="What is the capital of Pakistan?")
]

result = llm.invoke(messages)

print(result.content)