from langchain.text_splitter import CharacterTextSplitter

with open('D:\pretrained models\input.txt') as f:
    poem= f.read()

'''
reference: https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
           https://python.langchain.com/en/latest/modules/indexes/text_splitters/getting_started.html

length_function: how the length of chunks is calculated. Defaults to just counting number of characters,
but it’s pretty common to pass a token counter here.

chunk_size: the maximum size of your chunks (as measured by the length function).

chunk_overlap: the maximum overlap between chunks. It can be nice to have some overlap to maintain 
some continuity between chunks (eg do a sliding window).
'''
text_splitter= CharacterTextSplitter(
    separator="\n\n",
    chunk_size= 1000,
    chunk_overlap= 200,
    length_function= len,
)

texts= text_splitter.create_documents([poem])
print(texts[0])
print(len(texts))