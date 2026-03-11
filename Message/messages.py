from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# 1. Load Environment
load_dotenv()

# 2. Check Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint (Removed task="text-generation")
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)

# 4. CRITICAL FIX: Wrap the endpoint in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# 5. Define your messages
# (Note: You don't need to type "System: " or "Human: " inside the content! 
# LangChain handles the roles automatically.)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
]

try:
    # 6. CRITICAL FIX: Invoke chat_model, NOT llm
    result = chat_model.invoke(messages)
    
    # 7. Append the AI's response to your message history
    messages.append(AIMessage(content=result.content))
    
    # Print the AI's specific answer
    print(f"Chatbot: {result.content}\n")
    
    # Print the whole message history to verify it worked
    print("Full Conversation History:")
    for msg in messages:
        print(f"- {type(msg).__name__}: {msg.content}")

except Exception as e:
    print(f"An error occurred: {e}")