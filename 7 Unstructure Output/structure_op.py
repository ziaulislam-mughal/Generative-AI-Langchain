import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

# Load env
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)

# Schema
schema = [
    ResponseSchema(name="fact_1", description="A fact about topic"),
    ResponseSchema(name="fact_2", description="A fact about topic"),
    ResponseSchema(name="fact_3", description="A fact about topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "Black hole"})
print(result)