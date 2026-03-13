import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Load Environment
load_dotenv()

# check Token

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint
# 3. Setup the Endpoint
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)


# 4 . Wrap it in chathuggingface

model = ChatHuggingFace(llm=llm)

# jason parser


parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template="""
Give me the name, age, city of a fictional person.

{format_instruction}

Return ONLY valid JSON.
""",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = prompt_template | model | parser

result = chain.invoke({})
print(result)
