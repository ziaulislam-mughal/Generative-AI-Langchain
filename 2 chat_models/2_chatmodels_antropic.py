from langchain import AnthropicChat
from dotenv import load_dotenv 

load_dotenv() 


model = AnthropicChat(model="claude-2",
                      temperature=0.7,
                      max_tokens=100)

results  = model.invoke("What is the capital of France?") 
# output metedata + content 
print(results.content)