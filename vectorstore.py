import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader



# loader = PyPDFLoader("/turing.pdf")
# pages = loader.load_and_split()

loader = TextLoader('D:/pretrained models/chapter-5.txt')
documents = loader.load()

text_splitter= CharacterTextSplitter(
    separator="\n\n",
    chunk_size= 1000,
    chunk_overlap= 20,
    length_function= len,
)


docs= text_splitter.split_documents(documents)

# embeddingss = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings= HuggingFaceEmbeddings()


persist_directory= 'D:\pretrained models\persist_directory'

vector_db= Chroma.from_documents(docs, embeddings, persist_directory= persist_directory)
vector_db.persist()


retriever= vector_db.as_retriever(search_type= "mmr")
docs= retriever.get_relevant_documents("what is UDP?")

print(docs[0].page_content)

# sentence_embeddings = embeddingss.encode(pages[0])
# print(sentence_embeddings)
# print(len(str(texts)))
# print(len(str(doc_result)))
# print([texts])
