import os
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.utilities import SQLDatabase
from typing import Optional, Type
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from typing import List
#my methods
#from utils.db import append_csv_memory, csv_to_database

def add_to_csv(data, csv_path):
    if not os.path.exists(csv_path):
        data.to_csv(csv_path, header=True, index=False)
    else:
        data.to_csv(csv_path, mode='a', header=False, index=False)
map_topic = {'C':0,  'A': 1 , 'L':2 , 'CA':3, 'CL':4, 'AL':5, 'CAL':6,
              0:'C',  1:'A' , 2:'L' , 3:'CA',  4:'CL', 5:'AL', 6:'CAL'}

def append_to_csv(csv_path, data, save_type):
    if save_type == "question":
        columns = ['vid', 'question','gt','choices_list', 'attribution', 'topic',  'basis', 'related_times', 'related_person', 'related_location']
    else:
        columns = ['vid', 'question', 'gt','choices_list', 'attribution', 'topic', 'basis', 'related_times', 'related_person', 'related_location'] ## to do to revise

    df = pd.DataFrame(data, columns=columns)
    add_to_csv(df, csv_path)

def get_temporary_question(csv_path):
    if type(csv_path) == str:
        if not os.path.exists(csv_path):
            return "", 0
        df = pd.read_csv(csv_path)  
        df = df.drop_duplicates('question').reset_index(drop=True)
        df.to_csv(csv_path, header=True, index=False)
    else:
        df = csv_path
     
    questions_str = "\nUp to Now\n"
    type_dict = {'perception': [0,0,0,0,0,0,0], 'inference': [0,0,0,0,0,0,0]}
    for index, row in df.iterrows():
        type_dict[row['attribution']][map_topic[row['topic']]] += 1
        questions_str += f"Attribution: {row['attribution']}, Topic: {row['topic']}, Question: {row['question']}\n"
    questions_str += " have been saved. Carry on and don't generate duplicate questions.\n\n"

    perception_type = ""
    inference_type = ""
    number_enough = 0 
    each_type_num = 15
    for i in range(7):
        if type_dict['perception'][i] >= each_type_num:
            perception_type += f"Question of Attribution: perception, Topic: {map_topic[i]} is enough , please don't generate question about this type anymore!\n"
            number_enough += 1
        if type_dict['inference'][i] >= each_type_num:
            inference_type += f"Question of Attribution: inference, Topic: {map_topic[i]} is enough , please don't generate question about this type anymore!\n"
            number_enough += 1
        if type_dict['perception'][i] < each_type_num:
            perception_type  += "Question of Attribution: perception, topic: " + map_topic[i] + f", it's {type_dict['perception'][i]} questions only, generate more questions about this type please.\n"
        if type_dict['inference'][i] < each_type_num:
            inference_type  +="Question of Attribution: inference,  topic: " + map_topic[i] + f", it's {type_dict['inference'][i]} questions only, generate more questions about this type please.\n"
#    print("For perception:\n" + perception_type + "For inference:\n" + inference_type )
    questions_str += "For perception:\n" + perception_type + "For inference:\n" + inference_type + "\n### Don't only generate one question, think different please !!!" 
    return questions_str, number_enough

# define tool's input: question 
class questionInput(BaseModel):
    question: List[str] = Field(description="questions list you want to save.") 
    # to do
    choices: List[List[str]] = Field(description="choices list (4 choices) of each question. For each question, it should be [choice1, choice2, choice3, choice4].")
    gt: List[str] = Field(description="GroundTruth answer (actual answer, not 'A','B','C','D') for each question.") 
    attribution: List[str] = Field(description="attributions list of each question, use 'perception' or 'inference'.")
    topic: List[str] = Field(description="topics list related to each problem, C for Character, A for Action, L for Location. Choose from 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'.")
#    format: List[str] = Field(description="formats list of each question, 'matching' or 'multiple-choice")
    basis: List[str] = Field(description="inference basis list of each question, inference basis is the reason why we can get such a question's answer from video, it's a short description.")
    related_times: List[List[str]] = Field(description="time span list of video clips related to each question. For each question, it should be ['start1-end1', 'start2-end2',...]")
    related_person: List[List[str]] = Field(description="characters list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for character name from script.")
    related_location: List[List[str]] = Field(description="locations list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for location name from script.")


# define tool's input: question 
class CorssquestionInput(BaseModel):
    question: List[str] = Field(description="questions list you want to save.") 
    # to do
    choices: List[List[str]] = Field(description="choices list (4 choices) of each question. For each question, it should be [choice1, choice2, choice3, choice4].")
    gt: List[str] = Field(description="GroundTruth answer (actual answer, not 'A','B','C','D') for each question.") 
    attribution: List[str] = Field(description="attributions list of each question, use 'perception' or 'inference'.")
    topic: List[str] = Field(description="topics list related to each problem, C for Character, A for Action, L for Location. Choose from 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'.")
#    format: List[str] = Field(description="formats list of each question, 'matching' or 'multiple-choice")
    basis: List[str] = Field(description="inference basis list of each question, inference basis is the reason why we can get such a question's answer from video, it's a short description.")
    related_times: List[List[str]] = Field(description="time span list of video clips related to each question. For each question, it should be ['start1-end1', 'start2-end2',...]")
    related_person: List[List[str]] = Field(description="characters list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for character name from script.")
    related_location: List[List[str]] = Field(description="locations list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for location name from script.")
    


class saveQuestion(BaseTool):
    name: str = "Tools_of_saving_quetsion"
    description: str = "Follow the requirements of this tool to save questions !"
    args_schema: Type[BaseModel] = questionInput
    # private config
    csv_path: str = "csv/default.csv"                             # default csv
    video_id: str = "001.mp4"
    def __init__(self, csv_path: str, video_id: str):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        
    def _run(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              attribution: List[str], 
              topic: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if len(related_location) == 0 :
            related_location = [[] for i in range(len(question))]
        if len(related_person) == 0 :
            related_person = [[] for i in range(len(question))]
        data = [[self.video_id, 
                 question[i],
                 gt[i],
                 choices[i],
                 attribution[i],
                 topic[i],
                 basis[i], 
                 related_times[i],
                 related_person[i],
                 related_location[i]]
                 for i in range(len(question))
               ]
        # append to csv
        append_to_csv(self.csv_path, data, 'question')
        temporary_question_str, number_enough = get_temporary_question(self.csv_path)
        if number_enough ==14:
            return "questions have been enough. Stop generation!"
        return f"{temporary_question_str}"

    async def _arun(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              attribution: List[str], 
              topic: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")



class saveCorssQuestion(BaseTool):
    name: str  = "Tools_of_saving_cross_quetsion"
    description: str  = "useful when you generate cross episodes questions and need to save them"
    args_schema: Type[BaseModel] = questionInput
    # private config
    csv_path: str = "csv/default.csv"                             # default csv
    video_id: str = "001.mp4"
    def __init__(self, csv_path: str, video_id: List[str]):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        
    def _run(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              attribution: List[str], 
              topic: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if len(related_location) == 0 :
            related_location = [[] for i in range(len(question))]
        if len(related_person) == 0 :
            related_person = [[] for i in range(len(question))]
        data = [[self.video_id, 
                 question[i],
                 gt[i],
                 choices[i],
                 attribution[i],
                 topic[i],
                 basis[i], 
                 related_times[i],
                 related_person[i],
                 related_location[i]]
                 for i in range(len(question))
               ]
        # append to csv
        append_to_csv(self.csv_path, data, 'question')
        temporary_question_str, number_enough = get_temporary_question(self.csv_path)
        if number_enough ==14:
            return "questions have been enough. Stop generation!"
        return f"{temporary_question_str}"

    async def _arun(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              attribution: List[str], 
              topic: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")






#
#vid_name = "test"
#question_save_tool = saveCorssQuestion(csv_path=f"../csv_cross/{vid_name}.csv", video_id=['S01E01','S01E02','S01E03'])
#print(question_save_tool._run(['question'], [['a','b','c','D']],['a'],['perception'], ['A'], ['f'],[['00:11-01:22', '01:22-03:11']], [['character1', 'character2']], [['location1', 'location2']]))
#vid = "test"
#question_save_tool = saveQuestion(csv_path=f"csv/{vid}.csv", video_id=vid)
#print(question_save_tool._run('question', 'a', 't', 'f', 'b',[['00:01', '00:08'], ['00:11', '01:22']], [['character1', 'character2'], ['location1', 'location2']]))