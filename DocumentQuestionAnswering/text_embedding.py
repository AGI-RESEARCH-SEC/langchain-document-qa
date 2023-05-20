from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# ................................................................................................
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences= "This is a test sentence. This is 2ndtest sentences"
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings.shape)
# ..............................................................................................

# embeddings= HuggingFaceEmbeddings()
# text= "This is the test document"

# query_result= embeddings.embed_query(text)
# doc_result= embeddings.embed_documents([text])

# # print(doc_result)

# print(type(query_result))
# print(len(query_result))
# # print(query_result.shape())











# model2 = SentenceTransformer('all-MiniLM-L6-v2')
# #Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

# #Sentences are encoded by calling model.encode()
# embeddings = model2.encode(sentences)

# #Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")