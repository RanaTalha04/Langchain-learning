import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
load_dotenv()

embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id= "sentence-transformers/all-MiniLM-L6-v2",
    task= "feature-extraction"
)

text = "What is the capital of Pakistan?"

query_vector = embedding_model.embed_query(text=text)

documents = [
    "Islamabad is the capital city of Pakistan.",
    "Lahore is a major cultural hub.",
    "The currency used in Pakistan is the Rupee."
]

doc_embed = embedding_model.embed_documents(documents)

print(f"Query Vector length: {len(query_vector)}")
print(f"Sample values from vector: {query_vector[:5]}")