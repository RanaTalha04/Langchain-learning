import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm= llm)

st.header("Research Paper Summarizer")

prompt_input = st.selectbox("Select Research Paper Name: ", ["Attention is all you need", "Bert: Pretraining of Deep Bidirectional Transformers", "GPT-3: Language Models are few-shot learners"])

style_input = st.selectbox("Select Explaination Style: ", ["Beginner-friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explaination Length: ", ["Short (1-2 lines)", "Medium (4-5 lines)", "Long (15-20 lines)"])

template = load_prompt('template.json')

prompt = template.invoke({
    'prompt_input': prompt_input,
    'style_input': style_input,
    'length_input': length_input
})



if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
