from langchain_community.document_loaders import PyPDFLoader 

loader = PyPDFLoader(r'C:\Users\Administrator\Desktop\GenAI - Langchain\Gen AI\10 RAG - Document loaders\dl-curriculum.pdf')

docs = loader.load()

print(docs[0].page_content)
print(docs[1].metadata)