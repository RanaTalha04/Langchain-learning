from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Give me a catchy blog title name for {topic}",
    input_variables=['topic']
)

topic = input('Enter topic name: ')

formatted_prompt = prompt.format(topic=topic)

blog_title = model.invoke(formatted_prompt)

print(f"Generated blog title: {blog_title.content}")