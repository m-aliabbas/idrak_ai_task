"""
The **Query** will be a user query. It will interact with the Question Reterival System.
  This will be consits of followings:
  question(str): The Question Entered by User [This will be only user input]
  question_clean(str): Removing symbols from the question and making it lower-case. Making the question as same as input data
  similar_question(str): The question reterived from Reterival System that will be most likely to user question based on cosine similarity
  answer(str): Answer from the reterival question based on matched question
"""
"""
Importing required Modules
"""
import re #for regular experession. That will be used for text cleaning


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
