
"""**QuestionReterival** class is responisble for reading features database,extract features from
  query question. And then match query features with database using cosine similarity.The **constructor** of class, responsible for mapping class members with user parameters. Then it will load the model, model's wights
      feature database and tokenizer. And Initialize all related class members with them. 

**search_question function**

 This function is pivotal function. That is responisble for question reterival based on consine similarity.
    The working of function is following:
    **1:** tokenize the query questions using BertTokenizer 
    **2:** extract the features from tokenize questions
    **3:** find cosine similarity of query question with every entery of feature db
    **4:** getting the id of entery who have maximum similarity with query
    **5:** set the query answers, similarity score and similar questioon


"""
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
from models import FeaturesDBGenerator,FeatureExtractorBERTModel

class QuestionReterival:
  '''
  This class is responisble for reading features database,extract features from
  query question. And then match query features with database using cosine similarity.

  '''
  def __init__(self,features_db_path='question_answer_db.json',pretrain_base_path='sentence-transformers/all-MiniLM-L6-v2'\
               ,pretrain_model_weight_path='FE.pt'):
    try:
      '''
      The constructor of class, responsible for mapping class members with user parameters. Then it will load the model, model's wights
      feature database and tokenizer. And Initialize all related class members with them.

      arguments:
      --->features_db_path (str): It is path of JSON file have question,answers and features
      --->pretrain_base_path(str): Path of BertModel and BertTokenizer from Hugging Face Repository
      --->pretrain_model_weight_path(str): Path of model wights with those model is initialize/tunned when features database is made.
      It is a pt file. 
  

      '''
      self.features_db_path=features_db_path
      self.pretrain_base_path=pretrain_base_path
      self.pretrain_model_weight_path=pretrain_model_weight_path
      self.df=None
      self.features_db=None
      self.db_id=0
      self.similarity=None
      self.model=None
      self.tokenizer=None
      self.reload() #driver to load model in constructor. We can also use it when we update something with setter.
    except Exception as e:
      print(e)
  def read_db(self):
    '''
    reading the json database
    '''
    print('--- Reading Database ---')
    self.df=pd.read_json(self.features_db_path)
  def process_features(self): 
    '''
    This function make numpy arrays from JSON loaded dataframe 
    for further processing
    '''
    print('--- Processing Features ---')
    self.features_db=[np.array(i) for i in self.df.features.values] #get each feature vector and make numpy array of it and store in list
    self.features_db=np.array(self.features_db) #make numpy array of entire list. i.e. numpyndarray
  def load_tokenizer(self):
    '''
    Loading BertTokenizer for making token of text
    '''
    print('--- Loading Tokenizer ---')
    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_base_path)
  def load_model(self):
    '''
    Loading BertTokenizer for feature extraction of text
    '''
    print('--- Loading Model Structure ---')
    self.model=FeatureExtractorBERTModel(self.pretrain_base_path)
  def load_weights(self):
    '''
    Loading Model Weights that we computed when model is initialize/tunned for
    feature extraction. 
    '''
    print('--- Loading Pretrained Weights ---')
    weights = torch.load(self.pretrain_model_weight_path,map_location ='cpu') #Load model weights in pytorch format from path and map the tensors to cpu
    self.model.load_state_dict(weights) #load weights to model dict
    self.model.eval() #eval the model
  def reload(self):
    '''
    This is driver function which responisble for 
    reading database
    process features to numpy array
    load model,weights, & tokenizer
    '''
    self.read_db()
    self.process_features()
    self.load_tokenizer()
    self.load_model()
    self.load_weights()
  def show_db_head(self,n):
    '''
    This functions display the n numbers of features database items

    arguments:
    n(int): number of rows to display

    '''
    print(self.feature_db.head(n))
  def search_question(self,question=None):
    '''
    This function is pivotal function. That is responisble for question reterival based on consine similarity.
    The working of function is following:
    1: tokenize the query questions using BertTokenizer 
    2: extract the features from tokenize questions
    3: find cosine similarity of query question with every entery of feature db
    4: getting the id of entery who have maximum similarity with query
    5: set the query answers, similarity score and similar questioon
    '''
    try:
      encoded_input = self.tokenizer( question.get_question_clean(), padding=True, truncation=True, return_tensors='pt') #tokenize
      with torch.no_grad(): #no weights in update i.e. training
        model_output = self.model(**encoded_input) #compute features
      query_features=model_output.numpy() #tensor to numpy array
      self.similarity = cosine_similarity(query_features, self.features_db) #find cosine similarity
      self.db_id=np.argmax(self.similarity) #getting maximum number element/id from cosine similarity
      question.set_similarity_value(max(max(self.similarity))) #get the similarity of reterived id in similarity array
      question.set_similar_question(self.df.iloc[self.db_id]['Question']) #get the question at reterived id in db
      question.set_answer(self.df.iloc[self.db_id]['Answer']) ##get the answer at reterived id in db
    except Exception as e:
      print("Following Error Occured in Searching \n {}".format(e))
  #getter functions
  def get_features_db_path(self):
    return self.features_db_path
  def get_pretrain_base_path(self):
    return self.pretrain_base_path
  def get_pretrain_model_weight_path(self):
    return self.pretrain_model_weight_path

  #setter functions
  def set_features_db_path(self,features_db_path):
    self.features_db_path=features_db_path
  def set_pretrain_base_path(self,pretrain_base_path):
    self.pretrain_base_path=pretrain_base_path
  def set_pretrain_model_weight_path(self,pretrain_model_weight_path):
    self.pretrain_model_weight_path=pretrain_model_weight_path

  #representation functions
  def __str__(self):
    return f"Question Reterival Class feature_db={self.features_db_path} , model = {self.pretrain_model_weight_path}"
  def __repr__(self):
    return f"QuestionReterival({self.features_db_path},{self.pretrain_base_path},{self.pretrain_model_weight_path})"
