import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from ctransformers.langchain import CTransformers
# from langchain.llms import Ctransformers
from ctransformers import AutoModelForCausalLM
from langchain.llms import GPT4All
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
# from langchain.llms import OpenLM
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
import os


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chatty App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')




def main():
    pdf= st.file_uploader("Upload your pdf", type='pdf')
    if pdf is not None:
        # folder_name= 'persist_directory'
        # persist_directory= 'D:\pretrained models\pdfQA\persist_directory'
        # current_directory= os.getcwd()
        # # Check if the folder exists
        # folder_path = os.path.join(current_directory, folder_name)

        pdf_reader= PdfReader(pdf)
        text= ""
        for page in pdf_reader.pages:
            text= text+ page.extract_text()

        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap= 200,
            length_function= len
        )
        chunks= text_splitter.split_text(text=text)

        
        store_name= pdf.name[:4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore= pickle.load(f)
            st.write('Embedding loaded form the disk')
        else:
            embeddings= HuggingFaceEmbeddings()

            # vector_db= Chroma.from_documents(chunks, embeddings, persist_directory= persist_directory)
            # if os.path.isdir(folder_path):
            #     print('folder exist....')
            # else:
            #     print('Creating new persist directory')
            #     vector_db.persist()
            
            vectorstore= FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:

                pickle.dump(vectorstore, f)
            st.write("New embeddings are created!!")

        query= st.text_input("Ask question about your pdf files...")
        if query:
            docs= vectorstore.similarity_search(query=query, k=2)

            # repo_id= 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
            # repo_id= 'D:/pretrained models/generic functionality/models/ggml-model-gpt-2-117M.bin'
            repo_id= 'D:/pretrained models/generic functionality/models/ggml-gpt4all-l13b-snoozy.bin'

            # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.3, "max_length":140})
            # llm = CTransformers(model='marella/gpt-2-ggml')
            # llm = CTransformers(model='marella/gpt-2-ggml', callbacks=[StreamingStdOutCallbackHandler()])
            # llm= VertexAI()
            callbacks = [StreamingStdOutCallbackHandler()]
            llm = GPT4All(model=repo_id, callbacks=callbacks, verbose=True)
            # messages = [{"role": "user", "content": "Name 3 colors"}]
            # response= llm.chat_completion(messages)
            # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            # llm = LlamaCpp(model_path=repo_id, callback_manager=callback_manager, verbose=True)
            # llm= OpenLM("text-da-vinci-003")
            # llm = AutoModelForCausalLM.from_pretrained(repo_id, model_type='llama')
            # llm = AutoModelForCausalLM.from_pretrained(repo_id, model_type='llama')
            # llm = CTransformers(model=repo_id, model_type='gpt2')
            # llm = CTransformers(model='marella/gpt-2-ggml')

            # chain= load_qa_chain(llm=llm, chain_type="stuff")
            # response= chain.run(input_documents= docs, question=query)
            chain= load_summarize_chain(llm, chain_type="map_reduce")
            response= chain.run(docs)
            # response= llm("Give me complete code to train a two layer MLP neural network. ")

            st.write(response)
            st.write(docs)

if __name__=="__main__":
    main()