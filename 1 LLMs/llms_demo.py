from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI LLM

llm = OpenAI(model="gpt-3.5-turbo")
# Example usage
results = llm.invoke("What is the capital of France?")
print(results)