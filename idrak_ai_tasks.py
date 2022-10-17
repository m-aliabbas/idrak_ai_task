# -*- coding: utf-8 -*-
"""

This notebook contains the Bert based Question/Answer Searching system. The theme of notebook is following


1.   Data will be cleaned
2.   Bert Based Feature Extraction 
3.   Making Database of Bert Features
4.   Searching most Similar Question by Cosine Similarity of Querying Question. 
5.   Answering of Question

# Task1
## Cleaning/Preprocessing the Dataset

Importing required Modules
"""

import numpy as np  #for numarical processing
import pandas as pd #for dataframe processing
import re #for regular experession. That will be used for text cleaning
from sklearn.metrics.pairwise import cosine_similarity #for measuring cosien similarites

"""The **ProcessQuestionsDB** class will clean the input data and make a 
cleaned text database as output. The working of this class
based on pandas and regular expression.

**Construtor**:
constructor function of ProcessQuestionsDB class.
It will initialize the variables used. Also after initialization it will perform the processing operation by calling load_df and process_df functions. 


**pre-process**
This function will pre-process the database. For the cleaning operation an inner function cleanify will be called. After Cleaning it will Drop the rows which contains empty columns Also It will Drop the duplicate rows.

**cleanify**
This is inner function. It will first remove the unwantted symbols from text using regular expression. Then Keep the numbers, alphabets, and question mark .

**load_df**
    This function will load the dataframe from input_df_path
    After Loading the Dataframe it will get the Question.

**Getter and Setter** For getting and setting differnt data
"""

class ProcessQuestionsDB:
  '''
  The Python class will clean the input data and make a 
  cleaned text database as output. The working of this class
  based on pandas and regular expression.

  '''
  def __init__(self,input_df_path='S08_question_answer_pairs.txt',\
               output_df_path='idrak_ai_qa_pairs.csv',seprator='\t',question_col='Question',\
               answer_col='Answer'):
    '''
    This function is constructor function of ProcessQuestionsDB class.
    It will initialize the variables used. Also after initialization it
    will perform the processing operation by calling load_df and process_df 
    functions. 
    -----> Parameters: 
    -------> input_df_path(str): The Path of Input Question Database/csv/txt. (The input file should be txt/csv)
    -------> output_df_path(str): The Path where processed Database will be saved ('The address of ouput csv')
    -------> seprator(str): The Items of CSVs are seprate by delimiter; which is mostly Comma(,) or Tab/set of spaces (\t).
    -------> question_col(str): The name  of column in your database which contains Questions.
    -------> answer_col(str): The name of Answer containing column in your database


    '''

    self.input_df_path=input_df_path
    self.seprator=seprator
    self.question_col=question_col
    self.answer_col=answer_col
    self.output_df_path=output_df_path
    self.df=pd.DataFrame() #creating an empty dataframe 
    self.load_df() #calling load_df method for reading dataframe
    self.process_df() #cleaning the dataframe and saving output csv
  def load_df(self):

    '''
    This function will load the dataframe from input_df_path
    After Loading the Dataframe it will get the Question/Answer Columns from the Database
    '''

    df=pd.read_csv(self.input_df_path,sep=self.seprator) 
    self.df['Question']=df[self.question_col] 
    self.df['Answer']=df[self.answer_col]
  def process_df(self):
    '''
    This function will pre-process the database. For the cleaning operation
    an inner function cleanify will be called. 
    After Cleaning it will Drop the rows which contains empty columns,
    Also It will Drop the duplicate rows

    Output:
      ---> The dataframe containing three columns 
      -----> Question: The Question mentioned in Database
      -----> Answer: Answer stated in Database
      -----> text: The Cleaned Question
    '''

    def cleanify(text):
      '''
      This is inner function. It will first remove the unwantted symbols from text
      using regular expression. Then Keep the numbers, alphabets, and question mark 
      '''

      REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') #compile regulare expression for removing symbols
      BAD_SYMBOLS_RE = re.compile('[^0-9a-z ?]') #compile regulare expression to keep wanted data
      text = text.lower() #making text to lower case
      text = REPLACE_BY_SPACE_RE.sub(' ', text)  #applying 1 and 2nd mentioned re
      text = BAD_SYMBOLS_RE.sub('', text)
      return text
    self.df.dropna(inplace=True) #remove empty rows
    self.df['text'] = self.df['Question'].apply(cleanify) #applying the cleanify method
    self.df=self.df.drop_duplicates(subset=['text'], keep='last') #removing duplicated
    self.save_process_df() #saving the processed database


  def save_process_df(self):

    '''
    The function to save dataframe to output_df_path
    '''

    self.df.to_csv(self.output_df_path)
  
  #getter function
  def get_dataframe_head(self,n):

    '''
    The function to get to n rows of dataframe

    parameters:
    n (int): The number of rows to be selected
    
    return: dataframe
    '''

    return self.df.head(n)

  #setter functions
  def set_input_df_path(self,input_df_path):

    '''
    Setter  Method for Setting input_df_path

    Parameters:
    -----> input_df_path(str): The Path of Input Question Database/csv/txt. (The input file should be txt/csv)
    '''

    self.input_df_path=input_df_path
  def set_output_df_path(self,output_df_path):

    '''
    Setter  Method for Setting output_df_path

    Parameters:
    -----> output_df_path(str): The Path where processed Database will be saved ('The address of ouput csv')
    '''

    self.output_df_path=output_df_path
  def set_seprator(self,sep):

    '''
    Setter  Method for Setting delimiter

    Parameters:
    -----> output_df_path(str): The Items of CSVs are seprate by delimiter; which is mostly Comma(,)
     or Tab/set of spaces (\t).
    '''

    self.seprator=sep
  def set_question_col(self,q_col):
    
    '''
    Setter  Method for Setting delimiter

    Parameters:
    -----> question_col(str): The name  of column in your database which contains Questions.
    '''

    self.question_col=q_col
  def set_answer_col(self,a_col):
   
    '''
    Setter  Method for Setting delimiter

    Parameters:
    -----> answer_col(str): The name of Answer containing column in your database
    '''

    self.answer_col=a_col
    
  #representation functions
  def __str__(self):
    return f"This Questions Answer Processing Class. \n Input File: {self.input_df_path} \n Output File: {self.output_df_path}"
  def __repr__(self):
    return f"ProcessQuestionsDB({self.input_df_path},{self.output_df_path},{self.seprator},{self.question_col},{self.answer_col})"

"""Initialize an object of ProcessQuestionDB class, with given txt based question/answer database. Also define the seprator and cols for questions and answers"""

data_processor=ProcessQuestionsDB(input_df_path='S08_question_answer_pairs.txt',\
                output_df_path='idrak_ai_qa_pairs.csv',seprator='\t',question_col='Question',\
               answer_col='Answer')

"""# Task 2
## Defining Model for Feature Extraction. And Making Features DB

Installing required Libraries
"""


"""importing the modules"""

import torch 
from transformers import AutoTokenizer, AutoModel,BertModel #for Bert Tokenizer and Pretrained Bert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to check if GPU available else select CPU

import torch.nn.functional as F #Importing functional API
from torch import nn #Importing nn from torch that contains differnt layers

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

"""Defining the class Object of FeaturesDBGenerator Class. On Constructor it will do require job. And Save a JSON file. """

features_db_generator=FeaturesDBGenerator(dataset_path='idrak_ai_qa_pairs.csv',extractor_model_weight_path='FE.pt',output_db_path='qa_db.json')

"""#Task 3
## Doing Queries and getting similar question answers

The **Query** will be a user query. It will interact with the Question Reterival System.
  This will be consits of followings:
  question(str): The Question Entered by User [This will be only user input]
  question_clean(str): Removing symbols from the question and making it lower-case. Making the question as same as input data
  similar_question(str): The question reterived from Reterival System that will be most likely to user question based on cosine similarity
  answer(str): Answer from the reterival question based on matched question
"""

class Query:
  '''
  This class will be a user query. It will interact with the Question Reterival System.
  This will be consits of followings:
  question(str): The Question Entered by User [This will be only user input]
  question_clean(str): Removing symbols from the question and making it lower-case. Making the question as same as input data
  similar_question(str): The question reterived from Reterival System that will be most likely to user question based on cosine similarity
  answer(str): Answer from the reterival question based on matched question
  
  '''
  def __init__(self,question):
    '''
    The constructor will be initialize the parameters and get the user question. After getting user question it will clean it

    arguments:
    question(str): the user question string that will be passed to database
    '''
    self.__REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') #regular expression for removing unwanted symbol
    self.__BAD_SYMBOLS_RE = re.compile('[^0-9a-z ?]') #regular expression for keeping wanted text
    self.question=question
    self.question_clean=''
    self.similar_question=''
    self.similarity_value=0
    self.answer=''
    self.cleanify() #clean the text
  def cleanify(self):
    '''
    This function is cleaning the user question and assigning the cleaned question to question_clean
    '''
    text=self.question
    text = text.lower() #making lower case
    #applying re
    text = self.__REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = self.__BAD_SYMBOLS_RE.sub('', text)
    self.question_clean=text
  
  #Getter Functions

  def get_question(self):
    return self.question
  def get_question_clean(self):
    return self.question_clean
  def get_answer(self):
    return self.answer
  def get_similar_question(self):
    return self.similar_question
  def get_similarity_value(self):
    return self.similarity_value

  #Setter Functions

  def set_question(self,question):
    '''
    User can set the question text. This function will also set the text and clean it

    argument:

    question(str)
    '''
    self.question=question
    self.cleanify() #clean the question

  #These setter will be called by Question Reterival Class

  def set_answer(self,answer):
    self.answer=answer
  def set_similar_question(self,sim_que):
    self.similar_question=sim_que
  def set_similarity_value(self,sim_val):
    self.similarity_value=sim_val
  #representation functions
  def __repr__(self):
    return f"Query({self.question},{self.answer},{self.similar_question},{self.similarity_value:.4f})"
  def __str__(self):
    return f"Your Question: {self.question} \nSimilar Question {self.similar_question} \nAnswer {self.answer}"

"""Making an object of query class. And Passing it quering question."""

query=Query('When did Political career of lincoln start?') #defining a query object

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

"""Initializing the object of Question Reterival class and passing it a json based features question/answer database."""

qr=QuestionReterival(features_db_path='qa_db.json')

"""Searching the question from databse"""

qr.search_question(query)

"""Displaying the query"""

print(query)

"""Defining another question/query object"""

query1=Query('Start of Political Career of lincoln?')

"""searching the query"""

qr.search_question(query1)

print(query1)



"""## Thanks for reading.
Credit: Mohammad Ali Abbas (former machine learning developer SafeBeatRX)

@: maliabbas366@gmail.com
github: github.com/m-aliababs
"""

