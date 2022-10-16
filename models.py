
"""
Importing required Modules
"""
import numpy as np  #for numarical processing
import pandas as pd #for dataframe processing
import re #for regular experession. That will be used for text cleaning
from sklearn.metrics.pairwise import cosine_similarity #for measuring cosien similarites
import torch 
from transformers import AutoTokenizer, AutoModel,BertModel #for Bert Tokenizer and Pretrained Bert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to check if GPU available else select CPU

import torch.nn.functional as F #Importing functional API
from torch import nn #Importing nn from torch that contains differnt layers

"""# Task 2
## Defining Model for Feature Extraction. And Making Features DB
"""

"""**FeatureExtractorBERTModel** class is PyTorch based feature extractor class.
This class get a pretrain_base_path as argument. 
    The pretrain_base_path will be address of pretrained model
    from huggingface repo.
"""

class FeatureExtractorBERTModel(nn.Module):
    '''
    This Class is PyTorch based feature extractor class.
    This class get a pretrain_base_path as argument. 
    The pretrain_base_path will be address of pretrained model
    from huggingface repo.

    arguments:
    
    nn.Module: nn module of pytorch

    '''
    def __init__(self,pretrain_base_path):
          '''
          1) Getting pretrained Bert Model
          2) Adding a Linear Layer after BertPooler layer in bert i.e. after last layer.
          
          arguments:
          
          pretrain_base_path(str): The pretrain_base_path will be address of 
          pretrained model from huggingface repo.

          output: a tensor containing the features of input token of text

          '''
          super(FeatureExtractorBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained(pretrain_base_path) #pretrain bert head. 
          self.linear1 = nn.Linear(384, 256) #a linear layer with input shape of 384 and output shae of 256 as required
    def forward(self, **args):
          '''
          Processing the feature extractor part. 
          **args contain: input_ids: Tokenize text using BertTokenizer
                          attention_mask: Attention Mask of Text Using BertTokenizer
          '''
          bert_output = self.bert(
               args['input_ids'], 
               attention_mask=args['attention_mask']) #getting embedding/feature of text from Bert Model
          linear1_output = self.linear1(bert_output.last_hidden_state[:,0,:]\
                                        .view(-1,384)) #getting last BertPoolerLayer Output and passing to linear layer
          return linear1_output

"""**FeaturesDBGenerator** is responsible for making features database of Question/Answers for efficent Reterival of similar questions based on BertModel.

**Assumption**: 
  We are using `[sentence-transformers/all-MiniLM-L6-v2]` Bert Model. Because it is
  trained for sentecess similarity. Also it have less parameters.

The constructor of class have defaults values for arguments. The it will assign to class members.
    After the assignment of correct values it will do processing it self and make a JSON file containing
    the features and crossponding question/answers.
"""

class FeaturesDBGenerator:
  '''
  This class is responsible for making features database of Question/Answers
  for efficent Reterival of similar questions based on BertModel.

  Assumption: 
  We are using [sentence-transformers/all-MiniLM-L6-v2] Bert Model. Because it is
  trained for sentecess similarity. Also it have less parameters. 
  '''
  
  def __init__(self,pretrain_base_path='sentence-transformers/all-MiniLM-L6-v2',dataset_path=None,extractor_model_weight_path=None,output_db_path=None):
    '''
    The constructor of class have defaults values for arguments. The it will assign to class members.
    After the assignment of correct values it will do processing it self and make a JSON file containing
    the features and crossponding question/answers.

    arguments:

    pretrain_base_path(str): the hugging-face repo path of required Bert Model 
    dataset_path(str): path were processed question answer database is stored. The database
    -> sould contain three columns. 1. Question 2. Answer 3. text

    extractor_model_weight_path(str): Were weights of model stored which is used for feature extraction of Questions. Because we add a last layer
    and it will have random weights. So We need same extractor for making database and processing query string.

    output_db_path(str): string where we will save the output json database having 4 columns 1. Question 2. Answer 3. text 4. features
    '''

    try:
      self.pretrain_base_path=pretrain_base_path
      self.extractor_model_weight_path=extractor_model_weight_path
      self.dataset_path=dataset_path
      self.output_db_path=output_db_path
      self.tokenizer = None #tokenizer for making token of text initialy None will be define by generate_base_tokenizer()
      self.model=None # Bert Model for text features initialy None will be define by generate_base_model()
      self.df=None # database of processed questions initialy None will be define by read_question_db()
      self.process() #This function is driver function and its running in Constructor to define the above mention class members by loading/downloading/reading models&data
    except Exception as e:
      print(e)
  def process(self):
    '''
    This function is driver function and its running in Constructor to define the above mention class members by loading/downloading/reading models&data
    '''
    self.generate_base_tokenizer() 
    self.generate_base_model()
    self.read_question_db()
    self.extract_features()
    self.save_features_as_json()
    self.save_model_weights()
  def generate_base_model(self):

    '''
    Downloading the BertModel from Hugging-face repo.
    '''

    self.model=FeatureExtractorBERTModel(self.pretrain_base_path)
  def generate_base_tokenizer(self):
    '''
    Downloading the BertTokenizer from Hugging-face repo. Model and Tokenizer will be identicial
    '''
    self.tokenizer=AutoTokenizer.from_pretrained(self.pretrain_base_path)
  def read_question_db(self):
    '''
    reading the processed questions dataframe. The columns will be
    1. Question 2. Answer 3. text

    '''
    self.df=pd.read_csv(self.dataset_path)
  def extract_features(self):

    '''
    This is the main method for which the class is made. It is responsible for
    the job of feature extraction using Bert Model. 
    First it tokenize the text
    Pass text to Bert
    Save feature in dataframe
    '''

    print("--- Extracting Features ---")

    sentences=list(self.df['text'].values) #Getting all questions and making python list of them

    encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt') #Tokenize using Bert Tokenizer and returing data in Pytroch Tensor
    with torch.no_grad(): #We are not training
        model_output = self.model(**encoded_input) #pass the tokens to model and get model output/last layer output
    self.df['features']=list(model_output.numpy()) #convert the last layer output and save to dataframe column. Named features
    print("--- Features Extracted ---")

  def save_features_as_json(self):

    '''
    This function will save the features database in JSON format to desire path. That is define earlier.
    '''

    self.df.to_json(self.output_db_path)
    print("--- Features are Stored to {} ---".format(self.output_db_path))
  def save_model_weights(self):
    '''
    This function will save the weights in pt/pickle format. That will be use in inferene time.
    '''
    torch.save(self.model.state_dict(), self.extractor_model_weight_path) #getting model's wights and saving at desire path
    print("--- Model is saved to {} ---".format(self.extractor_model_weight_path))
  
  #the getter functions
  def get_model(self):
    '''
    return pytorch model
    '''
    return self.model
  def get_model_weights(self):
    '''
    return weights of pytorch model computed 
    '''
    return self.model.state_dict()

  #Setter Functions
  def set_pretrain_base_path(self,pretrain_base_path):
    self.pretrain_base_path=pretrain_base_path
  def set_output_db_path(self,output_db_path):
    self.output_db_path=output_db_path
  def set_extractor_model_weight_path(self,extractor_model_weight_path):
    self.extractor_model_weight_path=extractor_model_weight_path
  def set_dataset_path(self,dataset_path):
    self.dataset_path=dataset_path

  #functions for respresentation
  def __str__(self):
    return f'Feature Extracter and Model Saving Class'
  def __repr__(self):
    return f'FeaturesDBGenerator({self.dataset_path},{self.output_db_path} , {self.extractor_model_weight_path} )'
