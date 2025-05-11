from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the Groq LLM
llm = Groq(
    model="llama3-70b-8192",  # Model name must be specified
    api_key=os.getenv("GROQ_API_KEY"),  # Use 'api_key' instead of 'groq_api_key'
    temperature=0.3
)

# Use the 'complete' method instead of 'invoke'
result = llm.complete("What are AI agents?")
print(f"AI: {result.text}")