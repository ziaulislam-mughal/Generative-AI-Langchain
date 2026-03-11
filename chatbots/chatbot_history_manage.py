import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage , AIMessage  , SystemMessage

# 1. Load Environment 
load_dotenv()

# 2. Check Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint 

repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    task = "text-generation",
    max_new_tokens = 512,
    temperature = 0.7,
    huggingfacehub_api_token=token,
) 

# 4. Manage History 

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]


# 5. Wrap it in ChatHuggingFace (This handles the "conversational" format for you)
chat_model = ChatHuggingFace(llm=llm)

print(f"--- Chatbot Active (Model: {repo_id}) ---")

while True:
    input_user = input("You : ")
    chat_history.append(HumanMessage(content=input_user))
    if input_user.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chatbot. Goodbye!")
        break
    
    try:
        # Wrap the string in a HumanMessage for the Chat Model
        messages = [HumanMessage(content=input_user)]
        
        # Invoke the chat model
        response = chat_model.invoke(chat_history)
        chat_history.append(AIMessage(content = response.content))
        
        # For ChatHuggingFace, we DO use .content
        print("Chatbot :", response.content)
        
    except Exception as e:
        print(f"An error occurred: {e}")


print("--- Chat History ---")
print(chat_history)