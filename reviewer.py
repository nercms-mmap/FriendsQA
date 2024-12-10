import csv
import pandas as pd
import os
import argparse
import numpy as np
from utils.util import script_to_str, multi_script_to_str
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
#
#def getQuestion(csv_path):
#    Questions = "id, Question, Choices, groundtruth Answer, Related characters, Related locations"
#    question_csv = pd.read_csv(csv_path)
#    for i in range(len(question_csv)):
#        question = question_csv.loc[i,'question']
#        choice = question_csv.loc[i,'choices_list']
#        answer = question_csv.loc[i,'gt']
#        related_persons = question_csv.loc[i,'related_person']
#        related_locations = question_csv.loc[i,'related_location']
#        
#        Questions += f"{i+1}, {question}, {choice}, {answer}, {related_persons}, {related_locations}\n"
#    last_id = len(question_csv)
#    return last_id, Questions


def getQuestion(csv_path):
    Questions = "id, Question, Choices"
    question_csv = pd.read_csv(csv_path)
    for i in range(len(question_csv)):
        question = question_csv.loc[i,'question']
        choice = question_csv.loc[i,'choices_list']
        answer = question_csv.loc[i,'gt']        
        Questions += f"{i+1}, {question}, {choice}\n"
    last_id = len(question_csv)
    return last_id, Questions


def getPrompt(last_id, video_info, generate_questions):
    prompt =f"""You are very good at reviewing questions for correctness and answering given questions. Here is a video information for a movie or TV show, and the corresponding test questions for understanding the content of the movie or TV show. You are asked to evaluate each question, assessing its correctness (True or False). Assuming you are actually doing the test and can only watch the video,  you only need to give the corresponding assessment in the order in which they are presented. 

### Video Information
Please note that the video information contains the boundingbox, character, and line and line timestamps. The list of boundingboxes represents the top-left and bottom-right coordinate points, i.e. [top-left x, top-left y, bottom-right x, bottom-right y], and the time span of each boundingbox information is all the lines from the current boundingbox to the next one. If the attribute of a boundingbox is none, then please ignore this information and do not generate questions for this boundingbox. And the frame size is 1920*1080. video information are as follow:\n{video_info}

### Generated questions are as follow:
{generate_questions}

### Relevant Requirements
For correctness, There are two criteria for the correctness of a question and both of them should be satisified: 
1. The question must be relevant to the content of the video and can be answered with the only one Answer from Choices.
2. In the list of options for this question, there must be only one correct answer, with the other options unequivocally incorrect. 
You need to carefully review each question, ensuring that there are no factual errors, logical reasoning errors, etc., in the question and answer. For conservative,Please filter out the 20% of questions with low confidence in all questions by giving their correctness as False.


### Output Reuirements
Each line includes 3 elements, separated by spaces: id of the question, correctness of the question (True or False), the answer you want to use to answer this question (You can select from Choices.).

### output examples:
1 True XXX
2 True XXX
3 True XXX
4 True XXX
5 False XXX
...

Please give the reviewing result according to the id from 1 to {last_id}:
"""
    return prompt

def parser():
    parser = argparse.ArgumentParser("StoryMind's Reviewer", add_help=True)
    # GPT
    # parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    # parser.add_argument("--openai_proxy", type=str, help="proxy for chatgpt")
    # parser.add_argument("--model_name", default="gpt-4o",type=str, help="model name of GPT")
    # Gemini
    parser.add_argument("--gemini_model", default="gemini-1.5-pro-latest", type = str, help="model of gemini") 
    parser.add_argument("--google_api_key", required=True, type = str, help="api key of gemini")
    # Claude
    parser.add_argument('--claude_model', default="claude-3-5-sonnet-20240620", type = str, help="model of claude")  
    parser.add_argument('--claude_api_key', required=True, type = str, help="api key of claude")  
    parser.add_argument('--claude_proxy', type = str, help="api key of claude") 

    parser.add_argument("--episode", default = "single", type=str, help="'single' episode / 'cross' episode")
    return parser.parse_args()

    
def GPT4QA(text, api_key, base_url, model_type = 'gpt-4o'):
    client = OpenAI(base_url= base_url ,
                api_key=api_key,
                )
    PROMPT_MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user","content": text }
    ]
    params = {
        "model": model_type,
        "messages": PROMPT_MESSAGES,
        "max_tokens": 4096,
    }
    print(f"Waiting response from {model_type} ing：")
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


# prompt模版
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# llm问答函数
def llm_qa(llm, question, chat_history):
    chat_history.add_user_message(question)
    # llm处理链
    chain = prompt_template  | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "messages": chat_history.messages
        }
    )
    chat_history.add_ai_message(answer)
    return answer, chat_history

def split_list(input_list, n):
    output_list = []
    for i in range(0, len(input_list), n):
        output_list.append(input_list[i:i+n])
    return output_list



if __name__ == "__main__":
    args = parser()

    friends_dir = "scripts/"
    if args.episode == "single":
        file_list = sorted(os.listdir('csv'))
        for i, name in enumerate(file_list[:]):
            if not '.csv' in name:
                continue
            print(i,name)

            vid = name.split(".")[0]
            for model in ['gemini','claude']:
                if model == 'claude':
                    llm = ChatOpenAI(model_name=args.claude_model,                      # "claude-3-5-sonnet-20240620",
                                     openai_api_key=args.claude_api_key,               
                                     base_url= args.claude_proxy                        
                                    )
                    print("llm:",args.claude_model)
                elif model == 'gemini':
                    llm = ChatGoogleGenerativeAI(model=args.gemini_model,               # "gemini-1.5-pro-latest",
                                         google_api_key = args.google_api_key   
                                        )
                    print("llm:",args.gemini_model)
                output_path = f'check/{model}/{vid}.txt'
                if os.path.exists(output_path):
                    continue
                if os.path.exists(os.path.join(friends_dir, vid+".xlsx")):
                    script_path = os.path.join(friends_dir, vid+".xlsx")
                else:
                    script_path = os.path.join(friends_dir, vid+".xltx")

                questions_path = os.path.join('csv', name)
                last_id, questions = getQuestion(questions_path)
                video_info = script_to_str(script_path)
                prompt = getPrompt(last_id, video_info, questions)
                # 初始化一个聊天历史记录对象
                chat_history = ChatMessageHistory()
                # 将初始化好的llm，问题，聊天历史输入接口，即可返回：答案和新的聊天历史
                answer, chat_history = llm_qa(llm, prompt, chat_history)
                with open(output_path, 'w') as f:
                    f.write(answer)
    else:
        scripts_name_list = []
        for i, name in enumerate(sorted(os.listdir(friends_dir))):
            if 'ori' in name:
                continue
            scripts_name_list.append(os.path.join(friends_dir, name))      # (name) 
        split_list = split_list(scripts_name_list, 4)
        for i, group in enumerate(split_list):
            print(i, group)
            # scripts with too much bbox may contradict with gemini's policy, so it may need to modify the video_info manually , sometimes use boundingbox=False
            video_info = multi_script_to_str(group,  boundingbox = False)
            vids = [path.split("/")[-1][7:-5] for path in group]
            vid_name = vids[0] + "-" +  vids[-1]
            csv_path=f"csv_cross/{vid_name}.csv"
            last_id, questions = getQuestion(csv_path)
            for model in ['gemini','claude']:
                if model == 'claude':
                    llm = ChatOpenAI(model_name=args.claude_model,                      # "claude-3-5-sonnet-20240620",
                                     openai_api_key=args.claude_api_key,                
                                     base_url= args.claude_proxy                        
                                    )
                    print("llm:",args.claude_model)
                elif model == 'gemini':
                    llm = ChatGoogleGenerativeAI(model=args.gemini_model,               # "gemini-1.5-pro-latest",
                                         google_api_key = args.google_api_key   
                                        )
                    print("llm:",args.gemini_model)
                output_path = f'check_cross/{model}/{vid_name}.txt'
                if os.path.exists(output_path):
                    continue
                prompt = getPrompt(last_id, video_info, questions)
                # 初始化一个聊天历史记录对象
                chat_history = ChatMessageHistory()
                # 将初始化好的llm，问题，聊天历史输入接口，即可返回：答案和新的聊天历史
                answer, chat_history = llm_qa(llm, prompt, chat_history)
                with open(output_path, 'w') as f:
                    f.write(answer)
                