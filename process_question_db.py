# Task1
## Cleaning/Preprocessing the Dataset
"""
Importing required Modules
"""

import numpy as np  #for numarical processing
import pandas as pd #for dataframe processing
import re #for regular experession. That will be used for text cleaning

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