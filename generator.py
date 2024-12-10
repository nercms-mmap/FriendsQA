import os
# 设置环境变量 TOKENIZERS_PARALLELISM 为 "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from pathlib import Path
from pprint import pprint
import argparse
import tiktoken
import openai
import pandas as pd

#导入langchain的stool
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
#from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain import hub

from utils.tools import saveQuestion, saveCorssQuestion, get_temporary_question 
from utils.util import script_to_str, multi_script_to_str

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
import traceback
    
def parser():
    parser = argparse.ArgumentParser("StoryMind's generator", add_help=True)
    parser.add_argument("--gemini_model", default="gemini-1.5-pro-latest", type = str, help="model of gemini") 
    parser.add_argument("--google_api_key", required=True, type = str, help="api key of gemini") 
    parser.add_argument("--num_workers", default=4, required=True, type=int, help="number of workers") # workers num
    parser.add_argument("--worker", required=True, type=int, help="worker's order of agent") # choose one from a range from 0 to 4
    parser.add_argument("--begin", required=True, type=int, help="begin script of agent") # choose one from a range from 0 to 4
    parser.add_argument("--end", default= 999, type=int, help="end script of agent") # choose one from a range from 0 to 4
    parser.add_argument("--episode", default = "single", type=str, help="'single' episode / 'cross' episode")
    return parser.parse_args()

def extend_history(chat_history, prompt, answer):
    chat_history.extend(
        [
            HumanMessage(content=prompt),
            AIMessage(content=answer),
        ]
    )
    return chat_history

def get_agent_with_tools(vid, llm):
    question_save_tool = saveQuestion(csv_path=f"csv/{vid}.csv", video_id=vid)
    tools = [question_save_tool]
    prompt_react = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt_react)
    agent_executor= AgentExecutor(agent=agent,tools=tools,verbose=True,max_iteractions = 40,handle_parsing_errors=True)
#    agent_executor = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
#                                      max_iteractions = 40,
#                                      handle_parsing_errors=True,
#							     agent_kwargs = {
#							        "input_variables": ["input", "agent_scratchpad"] # "chat_history"
#							     }
#                                     )
    return agent_executor



def get_agent_with_tools_cross(vid, llm):
    vid_name = vid[0] + "-" +  vid[-1]
    question_save_tool = saveCorssQuestion(csv_path=f"csv_cross/{vid_name}.csv", video_id=vid)
    tools = [question_save_tool]
    prompt_react = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt_react)
    agent_executor= AgentExecutor(agent=agent,tools=tools,verbose=True,max_iteractions = 40,handle_parsing_errors=True)
#    agent_executor = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
#                                      max_iteractions = 40,
#                                      handle_parsing_errors=True,
#							     agent_kwargs = {
#							        "input_variables": ["input", "agent_scratchpad"] # "chat_history"
#							     }
#                                     )
    return agent_executor

def build_few_shots_examples(shots_txt="examples/single.txt"):
    with open(shots_txt, 'r') as f:
        shots_text = f.read()
    return shots_text
    
def prompt_engine(generated_questions ,video_info, shots_txt="examples/single.txt", prompt_type="qg"):
    prompt = "System: You are an expert in long story video comprehension and now need to put your students to the test by coming up with a series of questions for them to answer."
    if prompt_type == "qg":
        prompt += "Question for story video comprehension is divided into 2 attributions, 7 topics."
        prompt += """Attributions: 'perception', 'inference'.
Description: For questions of 'perception', it can be obtained from the appearance of the video directly, while questions of 'inference' need to analyze the content of the video and logical reasoning to get the results. Therefore, you need to focus on a more long-term understanding of the scripts, such as cross scene, even cross the whole script, generate less question about short-term understanding.\n"""
        prompt += """Topics: 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'
Description: We focuses on three core elements of the video: character, action, and location, where C stands for character, A stands for action, and L stands for location. Therefore, there are 7 possible topics, i.e., 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'. It should be attention that there should be only core elements in questions of topic. For example, 'C' can only involve 'character', not anything about 'action' or 'location'. You can only generate question about 'character'.\n"""

        prompt += """\nIn order to better validate the question, Please generate the basis of inference, a list of the start and end times of the relevant clips of the video(The span between start and end should be at least 30 seconds because short period contains too less information.) and a list of the relevant characters related to question."""
        
        
        prompt += f"\n\nPlease note that the video information contains the boundingbox, character, and line and line timestamps. The list of boundingboxes represents the top-left and bottom-right coordinate points, i.e. [top-left x, top-left y, bottom-right x,bottom-right y], and the time span of each boundingbox information is all the lines from the current boundingbox to the next one. If the attribute of a boundingbox is none, then please ignore this information and do not generate questions for this boundingbox. And the frame size is 1920*1080. video information are as follow:\n{video_info}\n"
        
        shots_text = build_few_shots_examples(shots_txt)
        prompt += f"\nQuestions Examples:\n{shots_text}\n\n"
        if generated_questions:
            prompt += generated_questions

        prompt += f"""Requirements: I hope relevant segments can be cross video clips. I want to generate questions of different difficulty, so for each type you need to help me generate questions of different difficulty by either focusing on more long-term video cip, or focusing on more complex character's relationship. You can even generate questions cover the whole episode. And the generation of questions should ideally be spread throughout the video rather than concentrated in certain scenes. In order to better validate the question, Please generate the basis of inference, a list of the start and end times of the relevant segments in the video, and a list of the relevant characters related to question.

Please generate questions with GroundTruth and choices list in multiple-choice format. For each 2 attributions and 7 topics, i.e., for each of the 14 type, you should generate at least 10 questions and questions should be with different degree of diffculty. You are specialize in using 'Tools_of_saving_quetsion' to save questions and you cane generate about 14 questions and save. Use tool 'Tools_of_saving_quetsion' to save question informations.
"""
        return prompt
    else:
        return "To do."




def prompt_engine_cross(generated_questions ,video_info, shots_txt="examples/cross.txt", prompt_type="qg"):
    prompt = "System: You are an expert in long story video comprehension and now need to put your students to the test by coming up with a series of questions for them to answer."
    if prompt_type == "qg":
        prompt += "Question for story video comprehension is divided into 2 attributions, 7 topics."
        prompt += """Attributions: 'perception', 'inference'.
Description: For questions of 'perception', it can be obtained from the appearance of the video directly, while questions of 'inference' need to analyze the content of the video and logical reasoning to get the results. Therefore, you need to focus on a more long-term understanding of the scripts, such as relationship, personality, emotions, etc. generate less question about short-term understanding.\n"""
        prompt += """Topics: 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'
Description: We focuses on three core elements of the video: character, action, and location, where C stands for character, A stands for action, and L stands for location. Therefore, there are 7 possible topics, i.e., 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'. It should be attention that there should be only core elements in questions of topic. For example, 'C' can only involve 'character', not anything about 'action' or 'location'. You can only generate question about 'character'.\n"""
        prompt += """\nIn order to better validate the question, Please generate the basis of inference, a list of the start and end times of the relevant clips of the video(The span between start and end should cross episode because the scripts are concated with multiple episodes of TV videos.) and a list of the relevant characters related to question."""
        
        prompt += f"\n\nPlease note that the video information contains the boundingbox, character, and line and line timestamps. The list of boundingboxes represents the top-left and bottom-right coordinate points, i.e. [top-left x, top-left y, bottom-right x,bottom-right y], and the time span of each boundingbox information is all the lines from the current boundingbox to the next one. If the attribute of a boundingbox is none, then please ignore this information and do not generate questions for this boundingbox. And the frame size is 1920*1080. video information are as follow:\n{video_info}\n"
        
        shots_text = build_few_shots_examples(shots_txt)
        prompt += f"\nQuestions Examples (each questions cover long range of time):\n{shots_text}\n\n"

        prompt += f"""Requirements: I hope relevant segments can be cross video clips. I want to generate questions of different difficulty, so for each type you need to help me generate questions of different difficulty by either focusing on more long-term video cip, or focusing on more complex character's relationship. You can even generate questions cover the whole episode. And the generation of questions should ideally be spread throughout the video rather than concentrated in certain scenes. In order to better validate the question, Please generate the basis of inference, a list of the start and end times of the relevant segments in the video, and a list of the relevant characters related to question.

Please generate questions with GroundTruth and choices list in multiple-choice format. For each 2 attributions and 7 topics, i.e., for each of the 14 type, you should generate at least 15 questions and questions should be with different degree of diffculty. You are specialize in using 'Tools_of_saving_cross_quetsion' to save questions and you must generate about 14 questions and save. Use tool 'Tools_of_saving_cross_quetsion' I give to save question informations.
"""
        if generated_questions:
            prompt += generated_questions
        # Use tool 'Tools_of_saving_quetsion' to save question informations
        return prompt
    else:
        return "To do."


def split_list(input_list, n):
    output_list = []
    for i in range(0, len(input_list), n):
        output_list.append(input_list[i:i+n])
    return output_list




if __name__=="__main__":
    args = parser()
    # To use model
    llm = ChatGoogleGenerativeAI(model=args.gemini_model, google_api_key = args.google_api_key)
    if args.episode == 'single':
        #encoding = tiktoken.encoding_for_model(args.model_name)
        max_iters = 50
        friends_dir = "scripts/"
        for i, name in enumerate(sorted(os.listdir(friends_dir))):
            if 'ori' in name:
                continue
            if i < args.begin:
                continue
            if i >= args.end:
                break
            if i % args.num_workers != args.worker:
                continue
            print(i, name)
            chat_history = []
            file_path = os.path.join(friends_dir, name)
            # some scripts may contradict with gemini's policy, so it may need to modify the video_info manually , sometimes use boundingbox=False 
            video_info = script_to_str(file_path)
            vid = name.split(".")[0]
            generated_questions, number_enough = get_temporary_question(f"csv/{vid}.csv")
            iters = 0
            while number_enough < 14:
                iters += 1
                print(f"#################################{iters}##################################################")
                agent_executor = get_agent_with_tools(vid, llm)
                prompt = prompt_engine(generated_questions, video_info, shots_txt="examples/single.txt", prompt_type="qg")
                if iters == 1:
                    with open("prompt.txt", "w") as file:
                        file.write(f"{vid}\n"+prompt+"\n\n")
                try:
                    answer = agent_executor.invoke({"input": prompt})
                except Exception as e:
                    error_message = traceback.format_exc()
#                     with open("log.txt", "a") as file:
#                        file.write(str(e))
#                        file.write("\n")
#                        file.write(error_message)
                generated_questions, number_enough = get_temporary_question(f"csv/{vid}.csv")
                if iters >= max_iters:
                    break
    else: # cross-episode
        max_iters = 50
        friends_dir = "scripts/"
        scripts_name_list = [] 
        for i, name in enumerate(sorted(os.listdir(friends_dir))):
            if 'ori' in name:
                continue
            scripts_name_list.append(os.path.join(friends_dir, name))      # (name) 
        split_list = split_list(scripts_name_list, 4)
        for i, group in enumerate(split_list):
            print(i, group)
            if i < args.begin:
                continue
            if i >= args.end:
                break
            if i % 4 != args.worker:
                continue
            vid = [path.split("/")[-1][7:-5] for path in group]
            vid_name = vid[0] + "-" +  vid[-1]
            generated_questions, number_enough = get_temporary_question(f"csv_cross/{vid_name}.csv")
            # scripts with too much bbox may contradict with gemini's policy, so it may need to modify the video_info manually , use boundingbox=False can avoid this
            video_info = multi_script_to_str(group,  boundingbox = False)
            prompt = prompt_engine_cross(generated_questions ,video_info, shots_txt="examples/cross.txt", prompt_type="qg") 
            iters = 0
            while number_enough < 14:
                iters += 1
                print(f"#################################{iters}##################################################")
                agent_executor =  get_agent_with_tools_cross(vid, llm)
                prompt = prompt_engine(generated_questions, video_info, shots_txt="examples/cross.txt", prompt_type="qg")
                if iters == 1:
                    with open("prompt.txt", "w") as file:
                        file.write(f"{vid_name}\n"+prompt+"\n\n")
                try:
                    answer = agent_executor.invoke({"input": prompt})
                except Exception as e:
                    error_message = traceback.format_exc()
                    # 将异常信息写入log.txt文件
#                    with open("log.txt", "a") as file:
#                        file.write(str(e))
#                        file.write("\n")
#                        file.write(error_message)
                generated_questions, number_enough = get_temporary_question(f"csv_cross/{vid_name}.csv")
                if iters >= max_iters:
                    break
                
    