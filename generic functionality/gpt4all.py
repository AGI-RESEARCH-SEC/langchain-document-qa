from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler


template = """Question: {question} 

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = './models/ggml-gpt4all-j-v1.3-groovy.bin'  # replace with your desired local file path

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)


llm_chain = LLMChain(prompt=prompt, llm=llm)


question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)






