# Notice we only need this one import now!
from langchain_core.prompts import ChatPromptTemplate

# 1. Create the template using tuples
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain}."),
    ("human", "Explain in simple terms {topic}")
])

# 2. Fill in the variables
prompt = chat_template.format_messages(
    domain="data scientist", 
    topic="what is a data scientist"
)

# 3. Look at the magic!
print(prompt)