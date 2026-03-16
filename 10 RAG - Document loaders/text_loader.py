import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# 1. Load Environment
load_dotenv()

# 2. Check Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)

# 4. Wrap it in ChatHuggingFace (This handles the "conversational" format for you)
model = ChatHuggingFace(llm=llm)

# 5. Create a Prompt Template

prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables = ['poem']
)

parser = StrOutputParser()

loader = TextLoader(r"C:\Users\Administrator\Desktop\GenAI - Langchain\Gen AI\10 RAG - Document loaders\cricket.txt", encoding="utf-8")

docs = loader.load()

print(type(docs))

print(docs[0].page_content)
print("-"*100)
print(docs[0].metadata)

model = prompt | model | parser
result = model.invoke({'poem':docs[0].page_content})
print(result)


