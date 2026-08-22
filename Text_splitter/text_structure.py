from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """

The afternoon sun cast long, dramatic shadows across the abandoned railway station, where rust and wild ivy had long since claimed the iron tracks. Nature was slowly erasing the heavy footprints of human industry, sending delicate green tendrils through the cracks in the concrete platforms and wrapping old wooden benches in a thick blanket of moss. A solitary crow perched atop a broken signal post, its dark feathers gleaming in the fading light as it surveyed the quiet landscape below. The air smelled faintly of old iron, dry dust, and the sweet fragrance of overgrown wildflowers that had sown themselves between the rotting wooden ties. It was a place caught suspended between two worlds, existing as a quiet monument to a bustling past that everyone else had forgotten.

As evening approached, a cool mist began to roll in from the adjacent valley, blanketing the hollows of the landscape in a soft, ethereal glow. The wind picked up, whispering through the skeletal remains of the old cargo depot and causing the loose corrugated tin roof to rattle with a rhythmic, ghostly click. Nearby, a shallow stream bubbled over smooth river stones, its constant rushing sound offering a lively contrast to the heavy silence dominating the station grounds. Darkness fell, and the first stars began to pierce through the twilight sky, casting a pale light over the ruins. The space transformed into a sanctuary for nocturnal creatures, which emerged to claim the night in a world where human presence had faded into memory.
"""

# loader = PyPDFLoader("data/Directory/Week 12.pdf")

# docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
)   

# result = splitter.split_documents(docs)
result = splitter.split_text(text)
# print(result[0].page_content)
print(result)