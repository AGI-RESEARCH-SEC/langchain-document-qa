from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline
from llama_index import Document
from llama_index import GPTVectorStoreIndex

class FlanLLM(LLM):
    model_name= "google/flan-t5-base"
    pipeline= pipeline("text2text-generation", model= model_name, device=0, model_kwargs= {"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop= None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
    
    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    
    def _llm_type(self):
        return "custom"
    

llm_predictor= LLMPredictor(llm= FlanLLM())

hfemb= HuggingFaceEmbeddings()
embed_model= LangchainEmbedding(hfemb)

print("...........................................................................")
print(embed_model)

text1= '''Understanding how the brain works is considered one of humanity’s
grand challenges. The quest has spawned dozens of national and
international initiatives, such as Europe’s Human Brain Project and the
International Brain Initiative. Tens of thousands of neuroscientists work in
dozens of specialties, in practically every country in the world, trying to
understand the brain. Although neuroscientists study the brains of different
animals and ask varied questions, the ultimate goal of neuroscience is to
learn how the human brain gives rise to human intelligence.
You might be surprised by my claim that the human brain remains a
mystery. Every year, new brain-related discoveries are announced, new
brain books are published, and researchers in related fields such as artificial
intelligence claim their creations are approaching the intelligence of, say, a
mouse or a cat. It would be easy to conclude from this that scientists have a
pretty good idea of how the brain works. But if you ask neuroscientists,
almost all of them would admit that we are still in the dark. We have
learned a tremendous amount of knowledge and facts about the brain, but
we have little understanding of how the whole thing works.
'''

text_list= [text1]

documents= [Document(t) for t in text_list]

# set number of output tokens
num_output = 200
# set maximum input size
max_input_size = 212
# set maximum chunk overlap
max_chunk_overlap = 15


prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# index = GPTListIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

response = index.query( "Whatthe author wants to say?")
response2= index.query("What is the book about?")

print(response)
print(response2)