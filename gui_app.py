from ctypes import alignment 
import flet
from flet import IconButton, Page, Row, TextField, icons,Text,ElevatedButton
from query import Query
from process_question_db import ProcessQuestionsDB
from question_reterival import QuestionReterival

def main(page: Page):
    '''
    This app is developed in FLET.
    A Python Based Flutter Module. 
    It is cross platform application
    '''
    page.title = "Idrak AI Question/Answer Reterival System"
    quest_ret=QuestionReterival(features_db_path='qa_db.json') #object of question reterival class
    txt_question = TextField(value="", text_align="right", width=500)
    title = Text(value="Idrak AI Question/Answer Reterival System",size=36,color="blue",weight="bold",italic=True,)
    input_label = Text(value="Enter Question?",size=15,color="green",weight="bold")
    similar_question_label= Text(value="Similar Question Found in DB: ",size=15,color="red",weight="bold")
    similar_question_text=Text(value="",size=15,color="green",italic=True)
    answer_label= Text(value="Answer: ",size=15,color="red",weight="bold")
    answer_text=Text(value="",size=12,color="green",italic=True)
    query=Query('') #object of Query Class we made
    def get_question(e):
        query_string = txt_question.value
        query.set_question(question=query_string)
        quest_ret.search_question(query)
        similar_question_text.value=query.get_similar_question()
        answer_text.value=query.get_answer()
        page.update()
    page.add(Row([title,],alignment="center"))
    page.add(
        Row(
            [
                input_label,
                txt_question,
                ElevatedButton("Search!", on_click=get_question),
            ],
            alignment="center"
        ),
        Row(
            [similar_question_label,similar_question_text]
        ),
        Row(
            [answer_label,answer_text]
        )
    )

flet.app(target=main)
