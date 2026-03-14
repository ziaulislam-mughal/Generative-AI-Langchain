import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda 
from pydantic import BaseModel , Field
from typing import Literal

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

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal["positive","negative"] = Field(description="Sentiment of the text")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# 5. Define Prompts
#  
prompt = PromptTemplate(
     template = """
Analyze the sentiment of the feedback.

Text: {feedback}

Return the answer strictly in this format:
{format_instructions}
"""
,
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = prompt | model | parser2



prompt_1 = PromptTemplate(
    template="write a kind words as a response if sentiment is postive \n {feedback}",
    input_variables=["feedback"]
)
prompt_2 = PromptTemplate(
    template="write a kind words as a response if sentiment is negative \n {feedback}",
    input_variables=["feedback"])

# 6. Build the Branching Chain 
runnacble_branch = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt_1 | model | parser),
    (lambda x:x.sentiment=='negative', prompt_2 | model | parser),
    RunnableLambda(lambda x:"Invalid Sentiment")
)

# 7. Combine into a Single Pipeline
final_chain = classifier_chain | runnacble_branch  
result = final_chain.invoke({'feedback':"you services are so bad . i am very disappointed, i give you zero star"})  
print(result)   

final_chain.get_graph().print_ascii()