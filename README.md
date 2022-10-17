# idrak_ai_task
This repo contains the task assign by idark ai. You can access the notebook from colab https://colab.research.google.com/drive/1r49W2dayUfltp7zQx1_4qY602IyLNn4F?usp=sharing or you can run locally idrak_ai_tasks (1).ipynb . The notebooks or python files contains all the details/documents and helps in comments and docs strings. 
Altough here we are sharing a brief details.
1. The question/answers text file will be process first. Empty and duplicate rows will be removed. Questions will be cleaned from unwanted symbols. ProcessQuestionDB class will be used for this purpose. 
``` 
data_processor=ProcessQuestionsDB(input_df_path='S08_question_answer_pairs.txt',\
                output_df_path='idrak_ai_qa_pairs.csv',seprator='\t',question_col='Question',\
               answer_col='Answer')
```
2. Features will be extracted using FeaturesDBGenerator class
```
features_db_generator=FeaturesDBGenerator(dataset_path='idrak_ai_qa_pairs.csv',extractor_model_weight_path='FE.pt',output_db_path='qa_db.json')
```
3. Query Object will be initialize that contain user query. And Answer to this object will assign by QuestionReterival system.
```
query=Query('When did Political career of lincoln start?') #defining a query object
```
4. QuestionReterival class is responisble for reading features database,extract features from
  query question. And then match query features with database using cosine similarity.
```
qr=QuestionReterival(features_db_path='qa_db.json')
```

5. Searching the query answer.
```
qr.search_question(query)
```
6. Displaying the answer
```
query.get_answer()
query.get_similar_question()
```
or
```
print(query)
```
You can use gui_aap.py 

# Ussage:
Idraak_Tutorial.mp4 this video contains all the process

## (Step 1) activate virtual envoirnment

## (Step 2) Installing require modules
```
pip install -r requirements.txt
```

## (Step 3) To extract features
```
python extract_feature.py
```
### Expected output
```
--- Extracting Features ---
--- Features Extracted ---
--- Features are Stored to qa_db.json ---
--- Model is saved to FE.pt ---

```

## (Step 4) Inference 
For inference we made a GUI based application. This application is developed in Flat. 
Flat is a python based module , which create flutter like applications.

```
python gui_app.py
```

It will run a flutter based application in which you will pass the question and it give us answers.

![Screenshot](screenshot.png)


Thanks 
