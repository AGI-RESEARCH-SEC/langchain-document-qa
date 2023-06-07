import langchain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# repo_id = "bigscience/bloom-560m"
# repo_id = "gpt2"
# repo_id = "distilgpt2"
# repo_id = "gpt2-medium"
# repo_id= "facebook/opt-1.3b"
# repo_id= "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
repo_id= "EleutherAI/pythia-70m"


# langchain.llm_cache= InMemoryCache()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":1.3, "max_length":240})


template = """Question: {question}

Answer: Let's think step by step.
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How to do habers process of manufacturing ammonia?"

print(llm_chain.run(question))