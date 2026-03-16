from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader


loader = DirectoryLoader(
    path = r'C:\Users\Administrator\Desktop\GenAI - Langchain\Gen AI\10 RAG - Document loaders\books',
    glob = '*.pdf', 
    loader_cls = PyPDFLoader
)


docs = loader.lazy_load()
for documents in docs:
    print(documents.metadata)
    print("-----------------------------")
    print("-----------------------------")
    print("Contect of page \n" + documents.page_content)
