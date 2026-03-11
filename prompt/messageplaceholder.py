from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage 

# create a template 

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Explain in simple terms {latest_question}"),
])


past_conversation = [
    SystemMessage(content="You are an expert data scientist."),
    HumanMessage(content="What is a data scientist?"),
    AIMessage(content="A data scientist is a professional who uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data."),
]

final_prompt = chat_template.format_messages(
    domain="data scientist",
    chat_history=past_conversation,
    latest_question="What is machine learning?"
)


print(final_prompt)
