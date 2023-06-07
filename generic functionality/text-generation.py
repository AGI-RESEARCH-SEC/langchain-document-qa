import langchain
from langchain import HuggingFaceTextGenInference
from langchain import HuggingFacePipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.cache import InMemoryCache


# repo_id= "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
langchain.llm_cache= InMemoryCache()
# llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", 
#                                         task="text-generation", model_kwargs={"temperature":1.2, "max_length":640})

llm = HuggingFacePipeline.from_model_id(model_id="OpenAssistant/pythia-12b-sft-v8-7k-steps", 
                                        task="text-generation", model_kwargs={"temperature":1.2, "max_length":220})

template= """
            Question: {question}
           
"""
prompt= PromptTemplate(template=template, input_variables=['question'])
llm_chain= LLMChain(prompt=prompt, llm=llm)

question= "How fMRI is used in brain scan?"

print(llm_chain.run(question))

