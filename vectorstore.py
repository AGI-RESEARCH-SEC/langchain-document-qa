import os
import getpass
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/turing.pdf")
pages = loader.load_and_split()


# text_splitter= CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size= 1000,
#     chunk_overlap= 200,
#     length_function= len,
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n"]
)


texts= text_splitter.create_documents([str(pages)])
# embeddingss = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings= HuggingFaceEmbeddings()
doc_result= embeddings.embed_documents([str(pages)])

# sentence_embeddings = embeddingss.encode(pages[0])
# print(sentence_embeddings)
print(len(str(texts)))
print(len(str(doc_result)))
print([texts])
# index_name = "langchain-demo"
# docsearch = Pinecone.from_documents([texts], doc_result, index_name=index_name)

# PINECONE_API_KEY = getpass.getpass('Pinecone API Key:')

# query = "What is the paper about?"
# docs = docsearch.similarity_search(query)