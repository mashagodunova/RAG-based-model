

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import tensorflow as tf
from torch.optim import AdamW
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from ray.data import ActorPoolStrategy
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
import haystack
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from llama_index import VectorStoreIndex, download_loader,  readers
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.embeddings import CustomEmbeddings
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
# from ftfy import fix_text
# from langchain.document_loaders import TextLoader
import pinecone
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# import sagemaker
# from sagemaker.huggingface import (
#     HuggingFaceModel,
#     get_huggingface_llm_image_uri
# )
from transformers import BertTokenizer
import random
import time
from langchain.document_loaders import TextLoader

loader = TextLoader('/content/constructicon_data.json')
documents = loader.load()
data = (list(list(documents[0])[0])[1])
data = json.loads(list(list(documents[0])[0])[1])
data['items'][0]['definitions'][2]['value']


pinecone.init(
	api_key='ea68a49b-80c1-42e9-8d31-aaffea875f7e',
	environment='gcp-starter'
)
index = pinecone.Index('retr')

model_name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
def generate_bert_embeddings(texts):
    embeddings = []
    for text in json.loads(list(list(texts[0])[0])[1])['items']:
        values = []
        if len(text['definitions'])>=3:
            values.append(text['definitions'][2]['value'])
        for v in text['examples']:
            values.append(v['value'])
        if len(values)>0:
            input_ids =tokenizer.encode(values, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            output = model(input_ids)
        embeddings.append(output.pooler_output)  # we can choose another type of output to write into the database
    return embeddings

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


bert_embeddings = generate_bert_embeddings(documents)

res = []
for text in json.loads(list(list(documents[0])[0])[1])['items']:
    values = " "
    if len(text['definitions'])>=3:
        values+=text['definitions'][2]['value']
    else:
        for v in text['examples']:
            values+=(v['value'])
    res.append(values)



count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(res), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(res))
    # get batch of lines and IDs
    lines_batch = res[i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    embeds = bert_embeddings[i].tolist()
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(list(to_upsert))





