from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Schema
class Review(BaseModel):
    
    key_theme: list[str] = Field(description="Write down all the key themes discussed in the review in a list.")
    summary: str = Field(description= "A breif summary of the review.")
    sentiment: Literal['Pos', 'Neg', 'Neut'] = Field(description="Return sentiment of the review; either Negative, Positive or Neutral.")
    pros: Optional[list[str]] = Field(description="Write down all the pros in a list.")
    cons: Optional[list[str]] = Field(description="Write down all the cons in a list.")
    name: Optional[str] = Field(description="Write the name of Reviewers.")
    
    
structured_model = model.with_structured_output(Review, method='json_mode')

prompt = ChatPromptTemplate([
    ('system', 'You are a helpful AI assistant. You must respond strictly in JSON format matching this shcema {schema}'),
    ('human', '{input}')
])

chain = prompt | structured_model

review_text = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
                                 
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Reviewer name: Muhammad Talha
"""

result = chain.invoke(
    {
        'schema': structured_model,
        'input': review_text
    }
)
print(result)
