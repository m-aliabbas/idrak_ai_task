"""
To run the application/inference you need a feature database. 
To create the feature database you need to pass a csv files having question_answer data.
We will pass S08_question_answer_pairs.txt to this module for extracting feature and process the data.
"""

from query import Query
from process_question_db import ProcessQuestionsDB
from question_reterival import QuestionReterival
from models import FeatureExtractorBERTModel,FeaturesDBGenerator


#Task 1:
## Data Processing and Making Cleaned Question answer pairs

process_question_db=ProcessQuestionsDB(input_df_path='S08_question_answer_pairs.txt',\
               output_df_path='idrak_ai_qa_pairs.csv',seprator='\t',question_col='Question',\
               answer_col='Answer')
#Task2
## Defining the BertModel and Extracting Features using BERT
"""
  FeaturesDBGenerator is responsible for making features database of Question/Answers
  for efficent Reterival of similar questions based on BertModel.

  Assumption: 
  We are using [sentence-transformers/all-MiniLM-L6-v2] Bert Model. Because it is
  trained for sentecess similarity. Also it have less parameters. 

    arguments:

    pretrain_base_path(str): the hugging-face repo path of required Bert Model 
    dataset_path(str): path were processed question answer database is stored. The database
    -> sould contain three columns. 1. Question 2. Answer 3. text

    extractor_model_weight_path(str): Were weights of model stored which is used for feature extraction of Questions. Because we add a last layer
    and it will have random weights. So We need same extractor for making database and processing query string.

    output_db_path(str): string where we will save the output json database having 4 columns 1. Question 2. Answer 3. text 4. features
    
"""
features_db_generator=FeaturesDBGenerator(pretrain_base_path='sentence-transformers/all-MiniLM-L6-v2',dataset_path='idrak_ai_qa_pairs.csv',extractor_model_weight_path='FE.pt',output_db_path='qa_db.json')
