from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, TextStreamer, BertTokenizer, BertModel
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
from torch import nn
from transformers import BertTokenizer

data = pd.read_csv('/content/drive/MyDrive/RAG_constr/parus_questions.csv')

data['var_1'] = data['input'].str.split(r' \n * ').str[1].str[2:]
data['var_2'] = data['input'].str.split(r' \n * ').str[2].str[2:]
data['question'] = data['input'].str.split(r' \n * ').str[0]
def shuffle_values(row):
    return pd.Series(np.random.permutation(row))

# Применение функции к каждой строке датафрейма
data[['var_1', 'var_2']] = data[['var_1', 'var_2']].apply(shuffle_values, axis=1)
data['answer'] = data.apply(lambda x: 1 if x['output'] == x['var_1'] else 0 if x['output'] == x['var_2'] else None, axis=1)
data.drop(columns=['Unnamed: 0', 'instruction', 'input', 'output'], inplace=True)


num_labels = 2
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
classifier = nn.Linear(768, num_labels)
model = nn.Sequential(model, classifier)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

X = data[['question','var_1','var_2']].values
y = data.answer.values

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    sents = []
    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        sents.append(sent.split())

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks, sents

"""Finding bigram with the biggest perplexity and replacing it"""

# def encode_sentence_with_bert(sentence, model, tokenizer):
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
model_name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sentence = data.question[20]
# Load the tokenizer and model




X = [' '.join(sublist) for sublist in X]
X_train, X_val, y_train, y_val =\
    train_test_split(X, y, test_size=0.1, random_state=2020)

MAX_LEN = 65
train_inputs, train_masks, train_sents = preprocessing_for_bert(X_train)
val_inputs, val_masks, val_sents = preprocessing_for_bert(X_val)

# res = inp['pooler_output'][0].tolist()
import pinecone

pinecone.init(
	api_key='ea68a49b-80c1-42e9-8d31-aaffea875f7e',
	environment='gcp-starter'
)
index = pinecone.Index('retr')

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
#batch_size = 16

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler)
#, batch_size=batch_size

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler)

class BertPerplexityClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super( BertPerplexityClassifier, self).__init__()
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


    def forward(self, input_ids, attention_mask, sentence):
        model_name = 'bert-base-multilingual-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)

        def encode_sentence_with_bert(sentence, model, tokenizer):
            # Tokenize the sentence

            tokens = sentence
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input__ids = torch.tensor([token_ids])

            # Get the model's predictions for the masked tokens
            outputs = self.bert(input_ids=input_ids)


            # Return the predictions
            return tokens, input__ids, outputs

        # Load the tokenizer and model
        #predictions = outputs[0]

        # Encode the sentence with BERT
        tokens, inputs, outputs = encode_sentence_with_bert(sentence, model, tokenizer)

        # Calculate the perplexity for each bigram
        rating = {}

        for i in range(len(tokens) - 1):
            if '[CLS]' not in tokens[i:i+2] and '[SEP]' not in tokens[i:i+2] and 'потому' not in tokens[i:i+2] and 'что' not in tokens[i:i+2]:
                bigram = tokens[i:i+2]
                bigram_ids = tokenizer.convert_tokens_to_ids(bigram)
                perplexity = torch.exp(outputs[0][0, i+1, -1]).item() # попробовать -1 вместо bigram_ids[-1]
                rating[f'{bigram}'] = perplexity

        # Sort the bigrams by perplexity
        sorted_rating = sorted(rating.items(), key=lambda item: item[1])
            # Tokenize the sentence
        max = sorted_rating[-1][0][2:-2].split('\', \'')
        predictions = outputs[0][0]
        #OR we can assert predictions = outputs
        # then we will go with predictions[0][:,word1,:] -- TEST THIS!!!


        def replace_value_in_vector(vector, token_index, new_value):
            vector_copy = vector.clone()
            vector_copy[token_index] = torch.tensor(new_value)
            return vector_copy




        word1 = tokens.index(max[0])
        word2 = tokens.index(max[1])
        f = predictions[word1].tolist()
        s = predictions[word2].tolist()
        bigram = []
        for i in range(len(f)):
            bigram.append((f[i] + s[i]) / 2)



        sim_pool = index.query(
            vector=bigram,
            top_k=25,
            include_values=True
          )['matches']
        matches = {}
        for vect in sim_pool:
          vector = []
          for item in vect['values']:
            vector.append(np.exp(-abs(item)))
          matches[vect['id']] = vector


        #bert_embedding, tokens = encode_sentence_with_bert(sentence, model, tokenizer)
        for item in max:
            python_token_index = tokens.index(f'{item}')
            new_value = list(matches.items())[-1][1]
            bert_embedding_with_replacement = replace_value_in_vector(predictions, python_token_index, new_value)

        cls = bert_embedding_with_replacement

        logits = self.classifier(cls[0].unsqueeze(0))

        return logits

from transformers import AdamW, get_linear_schedule_with_warmup

def initialize_rag_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertPerplexityClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

import random
import time
# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model."""
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for (step, batch), sent in zip(enumerate(train_dataloader), train_sents):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask, sent)
            #print(len(logits), len(b_labels))
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            print("\n")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set."""
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    for batch, sent in zip(val_dataloader, val_sents):
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask,sent)
            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())
            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

set_seed(42)    # Set seed for reproducibility
bert_rag_classifier, optimizer, scheduler = initialize_rag_model(epochs=2)
train(bert_rag_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)

