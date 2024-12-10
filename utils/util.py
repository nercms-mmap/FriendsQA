import pickle
import json
import os
from pathlib import Path
import argparse
import pandas as pd
import re
from pprint import pprint
import math

shot_time_path = "utils/shot_ins/shot_txt/"
shot_annotation_path = "utils/shot_ins/annotation/"
subtitle_time_path = "utils/shot_ins/scripts/"
wav_path = "utils/shot_ins/wav_vocal/"
bbox_path = "utils/shot_ins/data/"
season_episodes = [24, 24, 25, 24, 24, 25, 24, 24, 23, 17]
fps = 23.976
with open("utils/shot_ins/person_map.json") as p:
    c = p.read()
    person_map = json.loads(c)

def str2sec(x):
    """
    字符串时分秒转换成秒
    """
    # print(re.split(':|\.|,',x),x)
    h, m, s, ms = re.split(":|\.|,", x)

    return (
        int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    )  # int()函数转换成整数运算


def match_shot_subtitle(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    season_num = int(file_name[8:10])
    episode = int(file_name[11:13])
    episode_num = 0
    for i in range(season_num-1):
        episode_num += season_episodes[i]
    episode_num += episode
    with open(
                os.path.join(
                    bbox_path, "{}.json".format(episode_num)
                )
            ) as f:
                c = f.read()
                bbox = json.loads(c)
    shots_path = os.path.join(shot_time_path, str(episode_num) + ".txt")
    subtitle_path = str(episode_num) + ".xltx"
    if not os.path.exists(os.path.join(subtitle_time_path, subtitle_path)):
        subtitle_path = str(episode_num) + ".xlsx"
    shots_time = []
    subtitle_time = []
    with open(shots_path) as f:
        for index, line in enumerate(f.readlines()):
            shot_start_time, shot_end_time = line.split(" ")[0:2]
            shots_time.append(
                [
                    (float(shot_start_time) / fps) * 1000,
                    (float(shot_end_time) / fps) * 1000,
                ]
            )

    try:
        df = pd.read_excel(
            io=os.path.join(subtitle_time_path, subtitle_path),
            sheet_name=file_name,
            usecols=["start_time", "end_time"],
        )
    except:
        df = pd.read_excel(
            io=os.path.join(subtitle_time_path, subtitle_path),
            sheet_name=file_name + "_",
            usecols=["start_time", "end_time"],
        )
    for index, row in df.iterrows():

        subtitle_start_time = str2sec(row["start_time"])
        subtitle_end_time = str2sec(row["end_time"])
        subtitle_time.append([subtitle_start_time, subtitle_end_time])
    shot_num = 0
    for subtitle_num in range(len(subtitle_time)):
        # try:
        #     if len(subtitle_time[subtitle_num][2].split(" ")) < 8:
        #         continue
        # except:
        #     print(subtitle_time[subtitle_num], episode_num)
        #     continue
        while subtitle_time[subtitle_num][0] > shots_time[shot_num][1]:
            shot_num += 1
        if subtitle_time[subtitle_num][1] < shots_time[shot_num][0]:
            continue
        if (
            subtitle_time[subtitle_num][0] > shots_time[shot_num][0]
            and subtitle_time[subtitle_num][1] < shots_time[shot_num][1]
            or (
                subtitle_time[subtitle_num][1] > shots_time[shot_num][1]
                and (shots_time[shot_num][1] - subtitle_time[subtitle_num][0])
                / (subtitle_time[subtitle_num][1] - subtitle_time[subtitle_num][0])
                > 0.5
            )
        ):
            subtitle_time[subtitle_num].append(shot_num + 1)
    # print(subtitle_time)
    return subtitle_time, bbox
    

def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sheet_to_str(script_sheet, result_sheet,file_path, boundingbox = True):
#    sheet_str = "This is a script that follows the timeline.Each line we supply the character's name,location where the character locates and content including the character's dialogue and action.\n"
#    sheet_str += "characters - location - content\n"
    sheet_str = ""
    scene_id = 0
    last_target = 0
    dialog_shot, bbox = match_shot_subtitle(file_path)
    # print(dialog_shot,bbox)
    temp_shot = 0
    for i in range(len(result_sheet)):
        start = ":".join(result_sheet.iloc[i,1].strip()[:-4].split(":")[1:])
        end = ":".join(result_sheet.iloc[i,2].strip()[:-4].split(":")[1:])
        dialog = str(result_sheet.iloc[i,3]).strip()
        temporary_target = result_sheet.iloc[i,4]
        character = result_sheet.iloc[i,7].strip()
        location = result_sheet.iloc[i,8].replace(']', '')
        location = result_sheet.iloc[i,8].replace('.', '')

        if scene_id != result_sheet.iloc[i,6] and int(scene_id) < int(result_sheet.iloc[i,6]): # some script's scene index is wrong
            scene_id = result_sheet.iloc[i,6] 
            sheet_str += f"Scene: {scene_id}, Location: {location}\n"
        # redirect to script_sheet for more information
        if boundingbox:
            if i == 0:
                if len(dialog_shot[i])==2:
                    sheet_str += f"Boundingbox: none\n"
#                    print(sheet_str)
                    temp_shot = 0
                if len(dialog_shot[i])==3:
                    bbox_str = "Boundingbox: "
                    temp_shot = dialog_shot[i][2]
                    if str(temp_shot) in bbox.keys():
                        temp_bboxs = bbox[str(temp_shot)]
                        for person in temp_bboxs.keys():
                            person_name = person_map[person]
                            temp_bbox = temp_bboxs[person]
                            person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                            if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                                bbox_str = bbox_str + ""
                            else:
                                bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                        if bbox_str !="Boundingbox: ":
                            sheet_str += bbox_str.strip() + "\n"
            if temp_shot==0 and len(dialog_shot[i])==3:
                bbox_str = "Boundingbox: "
                temp_shot = dialog_shot[i][2]
                if str(temp_shot) in bbox.keys():
                    temp_bboxs = bbox[str(temp_shot)]
                    for person in temp_bboxs.keys():
                        person_name = person_map[person]
                        temp_bbox = temp_bboxs[person]
                        person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                        if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                            bbox_str = bbox_str + ""
                        else:
                            bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                    if bbox_str !="Boundingbox: ":
                        sheet_str += bbox_str.strip() + "\n"
            if temp_shot!=0 and len(dialog_shot[i])==2:
                sheet_str += f"Boundingbox: none\n"
                temp_shot = 0
        
            if temp_shot!=0 and len(dialog_shot[i])==3:
                if temp_shot!=dialog_shot[i][2]:
                    bbox_str = "Boundingbox: "
                    temp_shot = dialog_shot[i][2]
                    if str(temp_shot) in bbox.keys():
                        temp_bboxs = bbox[str(temp_shot)]
                        for person in temp_bboxs.keys():
                            person_name = person_map[person]
                            temp_bbox = temp_bboxs[person]
                            person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                            if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                                bbox_str = bbox_str + ""
                            else:
                                bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                        if bbox_str !="Boundingbox: ":
                            sheet_str += bbox_str.strip() + "\n"
        
        while last_target < temporary_target - 1 and temporary_target-last_target<10 and len(script_sheet)!=0:
            last_target += 1
            addition_info = script_sheet[script_sheet['index'] == last_target]
#            print(addition_info)
#            return
            if len(addition_info)==1:
                plot_type = addition_info.iloc[0,1].strip()
                plot_persons = addition_info.iloc[0,3]
                plot_content = str(addition_info.iloc[0,5]).strip()
                if plot_type == 'dialog':
                    dialog += str(addition_info.iloc[0,5]).strip()
                    continue
                else:
                    if type(plot_persons) == "str":
                        sheet_str += f"{plot_persons}-----"
                    sheet_str += f"{plot_content}\n"
          
        last_target = temporary_target if temporary_target > last_target else last_target
        
        sheet_str += f"({start}-{end}) {character}: {dialog}\n"
            
    return sheet_str

def script_to_str(file_path, boundingbox = True):
    script = pd.read_excel(file_path,sheet_name=None)
    names = list(script.keys())
    script_id = -1
    for i, name in enumerate(names):
        if "script" in name:
             script_id = i 
        if "result" in name:
             result_id = i
#    print(script_id, result_id)
    if script_id != -1:
        script_name = list(script.keys())[script_id]
        script_sheet = script[script_name]
    else:
        script_sheet = []
    result_name = list(script.keys())[result_id]
    result_sheet = script[result_name]
    script_str = sheet_to_str(script_sheet, result_sheet,file_path, boundingbox = boundingbox)
    return script_str

def time_remap(time, initial_time):
    split_time1 = initial_time.split(":")
    split_time2 = time.split(":")
    minute1, second1  = int(split_time1[0]), int(split_time1[1])
    minute2, second2  = int(split_time2[0]), int(split_time2[1])
    minute = minute1 + minute2 + (second1+second2)//60
    second = (second1 + second2)%60
    re_time = "{0:>02d}:{1:02d}".format(minute, second)
    return re_time
    
def sheet_to_str_ReMapTime(script_sheet, result_sheet,file_path, boundingbox = True, initial_time="00:00"):
#    sheet_str = "This is a script that follows the timeline.Each line we supply the character's name,location where the character locates and content including the character's dialogue and action.\n"
#    sheet_str += "characters - location - content\n"
    sheet_str = ""
    scene_id = 0
    last_target = 0
    dialog_shot, bbox = match_shot_subtitle(file_path)
    # print(dialog_shot,bbox)
    temp_shot = 0
    Up_to_now_time = initial_time
    for i in range(len(result_sheet)):
        start = ":".join(result_sheet.iloc[i,1].strip()[:-4].split(":")[1:])
        end = ":".join(result_sheet.iloc[i,2].strip()[:-4].split(":")[1:])
        dialog = str(result_sheet.iloc[i,3]).strip()
        temporary_target = result_sheet.iloc[i,4]
        character = result_sheet.iloc[i,7].strip()
        location = result_sheet.iloc[i,8].replace(']', '')
        location = result_sheet.iloc[i,8].replace('.', '')

        if scene_id != result_sheet.iloc[i,6] and int(scene_id) < int(result_sheet.iloc[i,6]): # some script's scene index is wrong
            scene_id = result_sheet.iloc[i,6] 
#            sheet_str += f"Scene: {scene_id}, Location: {location}\n"
            sheet_str += f"Location: {location}\n"
        # redirect to script_sheet for more information
        if boundingbox:
            if i == 0:
                if len(dialog_shot[i])==2:
                    sheet_str += f"Boundingbox: none\n"
#                    print(sheet_str)
                    temp_shot = 0
                if len(dialog_shot[i])==3:
                    bbox_str = "Boundingbox: "
                    temp_shot = dialog_shot[i][2]
                    if str(temp_shot) in bbox.keys():
                        temp_bboxs = bbox[str(temp_shot)]
                        for person in temp_bboxs.keys():
                            person_name = person_map[person]
                            temp_bbox = temp_bboxs[person]
                            person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                            if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                                bbox_str = bbox_str + ""
                            else:
                                bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                        if bbox_str !="Boundingbox: ":
                            sheet_str += bbox_str.strip() + "\n"
            if temp_shot==0 and len(dialog_shot[i])==3:
                bbox_str = "Boundingbox: "
                temp_shot = dialog_shot[i][2]
                if str(temp_shot) in bbox.keys():
                    temp_bboxs = bbox[str(temp_shot)]
                    for person in temp_bboxs.keys():
                        person_name = person_map[person]
                        temp_bbox = temp_bboxs[person]
                        person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                        if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                            bbox_str = bbox_str + ""
                        else:
                            bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                    if bbox_str !="Boundingbox: ":
                        sheet_str += bbox_str.strip() + "\n"
            if temp_shot!=0 and len(dialog_shot[i])==2:
                sheet_str += f"Boundingbox: none\n"
                temp_shot = 0
        
            if temp_shot!=0 and len(dialog_shot[i])==3:
                if temp_shot!=dialog_shot[i][2]:
                    bbox_str = "Boundingbox: "
                    temp_shot = dialog_shot[i][2]
                    if str(temp_shot) in bbox.keys():
                        temp_bboxs = bbox[str(temp_shot)]
                        for person in temp_bboxs.keys():
                            person_name = person_map[person]
                            temp_bbox = temp_bboxs[person]
                            person_bbox = temp_bbox[int(len(temp_bbox)/2)]
                            if person_bbox['w']<0 or person_bbox['h'] < 0 or person_bbox['y']<0 or person_bbox['y']<0 or person_bbox['x']< 0:
                                bbox_str = bbox_str + ""
                            else:
                                bbox_str = bbox_str + person_name + ": "+ f"[{person_bbox['w']}, {person_bbox['h']}, {person_bbox['y']}, {person_bbox['x']}] "
                        if bbox_str !="Boundingbox: ":
                            sheet_str += bbox_str.strip() + "\n"
        
        while last_target < temporary_target - 1 and temporary_target-last_target<10 and len(script_sheet)!=0:
            last_target += 1
            addition_info = script_sheet[script_sheet['index'] == last_target]
#            print(addition_info)
#            return
            if len(addition_info)==1:
                plot_type = addition_info.iloc[0,1].strip()
                plot_persons = addition_info.iloc[0,3]
                plot_content = str(addition_info.iloc[0,5]).strip()
                if plot_type == 'dialog':
                    dialog += str(addition_info.iloc[0,5]).strip()
                    continue
                else:
                    if type(plot_persons) == "str":
                        sheet_str += f"{plot_persons}-----"
                    sheet_str += f"{plot_content}\n"
          
        last_target = temporary_target if temporary_target > last_target else last_target
        start = time_remap(start, initial_time)
        end = time_remap(end, initial_time)       
        sheet_str += f"({start}-{end}) {character}: {dialog}\n"
        Up_to_now_time = end
    return sheet_str, Up_to_now_time


def multi_script_to_str(file_paths, boundingbox = True):
    script_str = ""
    initial_time="00:00"
    for num, file_path in enumerate(file_paths):
        script = pd.read_excel(file_path,sheet_name=None)
        names = list(script.keys())
        script_id = -1
        for i, name in enumerate(names):
            if "script" in name:
                 script_id = i 
            if "result" in name:
                 result_id = i
#        print(script_id, result_id)
        if script_id != -1:
            script_name = list(script.keys())[script_id]
            script_sheet = script[script_name]
        else:
            script_sheet = []
        result_name = list(script.keys())[result_id]
        result_sheet = script[result_name]
        one_script_string , initial_time = sheet_to_str_ReMapTime(script_sheet, result_sheet, file_path, boundingbox, initial_time)
        script_str += one_script_string
        if num != len(file_paths)-1:
            script_str +="\nOne episode end, followed by next episode.\n\n" 
        
    return script_str

if __name__=="__main__":
    with open("temp.txt",'w') as f:
        info = multi_script_to_str(["scripts/result_S09E15.xlsx",
	                               "scripts/result_S09E16.xltx",
	                               "scripts/result_S09E17.xlsx",
	                               "scripts/result_S09E18.xlsx"], boundingbox = True)
        f.write(info)
#    with open("temp.txt",'w') as f:
#        string = script_to_str("scripts/result_S01E02.xlsx", boundingbox =True)
#        f.write(string)

    
    