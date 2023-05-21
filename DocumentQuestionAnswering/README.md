## Document Question Answering 

Here I've used LLM to answer question about documents it was not trained on, on the idea of Retrieval augmented generation.

1. Load documents (using a Document Loader)
2. Split documents (using a Text Splitter)
3. Create embeddings for documents (using a Text Embedding Model)
4. Store documents and embeddings in a vectorstore

References: https://docs.langchain.com/docs/use-cases/qa-docs
https://python.langchain.com/en/latest/modules/indexes.html
