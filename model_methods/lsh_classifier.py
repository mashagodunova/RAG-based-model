
# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BertTokenizer, BertModel
# Import torch for datatype attributes
import torch
import tensorflow as tf
from torch.optim import AdamW
import numpy as np
#from llama_index import VectorStoreIndex, download_loader,  readers
#from langchain.embeddings import CustomEmbeddings
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
# from ftfy import fix_text
# from langchain.document_loaders import TextLoader
import pinecone
from torch import nn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# import sagemaker
# from sagemaker.huggingface import (
#     HuggingFaceModel,
#     get_huggingface_llm_image_uri
# )
import random
import time
import matplotlib.pyplot as plt
from random import shuffle



class BertLSHClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertLSHClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        sim_pool = index.query(
            vector=(outputs['pooler_output'][0].tolist()),
            top_k=40,
            include_values=True
          )['matches']
        matches = {}
        for vect in sim_pool:
          vector = []
          for item in vect['values']:
            vector.append(np.exp(-abs(item)))
          matches[vect['id']] = vector
        def create_hash_func(size: int):
          # function for creating the hash vector/function
          hash_ex = list(range(767))
          shuffle(hash_ex)
          return hash_ex

        def build_minhash_func(vocab_size: int, nbits: int):
          # function for building multiple minhash vectors
          hashes = []
          for _ in range(nbits):
              hashes.append(create_hash_func(vocab_size))
          return hashes
        minhash_func = build_minhash_func(20,767)
        def create_hash(vector):
        # use this function for creating our signatures (eg the matching)
            signature = []
            for func in minhash_func:
                for i in range(767):
                    idx = func.index(i)
                    if idx<len(vector):
                      signature_val = vector[idx]
                      if signature_val.any() >= 0.5:
                          signature.append(idx)
                          break
            return signature
        hash = {}
        for k,vect in matches.items():
          hash[k]=create_hash(vect)
        out = []
        for i in (outputs[0][:, 0, :]).detach().numpy():
          out.append(np.exp(-abs(i)))
        out = create_hash(out)
        def count_duplicates(list1, list2):
          return sum(1 for item in list1 if item in list2)
        sim = {}
        for k,h in hash.items():
          sim[k]=count_duplicates(out,h)
        max_key = max(sim, key=sim.get)
        for el in sim_pool:
          if el['id'] == max_key:
            final_vector = torch.tensor(el['values'])
        #[k]['values']
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        mean_val = last_hidden_state_cls + final_vector
        # Feed input to classifier to compute logits
        logits = self.classifier(mean_val)

        return logits

