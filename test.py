from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


embeddings= HuggingFaceEmbeddings()
persist_directory= 'D:\pretrained models\persist_directory'
# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever(search_type="mmr")

docs= retriever.get_relevant_documents("What is computer network?")

print(docs[0].page_content)