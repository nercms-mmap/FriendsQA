import os
import re
import json
import pandas as pd

from bisect import bisect_right, bisect_left
import numpy as np


friends_dir = "scripts/"


#def Video_base_info(script_name):
#    if os.path.exists(os.path.join(friends_dir, vid+".xlsx")):
#        script_path = os.path.join(friends_dir, vid+".xlsx")
#    else:
#        script_path = os.path.join(friends_dir, vid+".xltx")
#    script = pd.read_excel(script_path,sheet_name=None)
#    names = list(script.keys())
#    for i, name in enumerate(names):
#        if "script" in name:
#             script_id = i 
#        if "result" in name:
#             result_id = i
##    print(script_id, result_id)
#    result_name = list(script.keys())[result_id]
#    result_sheet = script[result_name] 
#    last_timestamp = result_sheet.iloc[len(result_sheet)-1, 2]
#    full_length = time_to_seconds(last_timestamp, "full")
#    scene_num = int(result_sheet.iloc[len(result_sheet)-1, 6])
#    return full_length, scene_num

def Useful_length(start_end_list):
    clip_length_list = []
    for start_end in start_end_list:
        if len(start_end) == 2:
            start = time_to_seconds(start_end[0])
            end = time_to_seconds(start_end[1])
            clip_length_list.append(end-start)
        else:
            clip_length_list.append(0)
    return sum(clip_length_list)
    



  
def add_to_csv(data, csv_path):
    if not os.path.exists(csv_path):
        data.to_csv(csv_path, header=True, index=False)
    else:
        data.to_csv(csv_path, mode='a', header=False, index=False)

def append_to_csv(csv_path, data, save_type):
    if save_type == "question_complexity":
        columns = ['vid', 'question', 'gt', 'choices_list', 'attribution', 'topic', 'basis', 'related_times', 'related_person', 'related_location','time_factors', 'content_factors']
    else:
        columns = ['question', 'attribution', 'topic', 'fromat', 'basis','times','time_complexity', 'related_instances', 'instance_complexity'] ## to do to revise

    df = pd.DataFrame(data, columns=columns)
    add_to_csv(df, csv_path)





def parse_script(result_sheet):
    last_timestamp = result_sheet.iloc[len(result_sheet)-1, 2]
    full_length = time_to_seconds(last_timestamp, "full")
    all_characters_list = []
    all_locations_list = []
    scene_time_list = [time_to_seconds(result_sheet.iloc[0, 1].strip(), 'full')]
    scene_id = 1
    last_start = '00:00:01'
    temporay_character = set()
    temporay_location = set()
    for i in range(len(result_sheet)):
        start = result_sheet.iloc[i, 1].strip()
        end = result_sheet.iloc[i, 2].strip()
        scene = result_sheet.iloc[i,6]
        characters = result_sheet.iloc[i,7].strip()
        location = result_sheet.iloc[i, 8].strip()
        # new_scene
        if scene_id != scene:
            scene_time_list.append(time_to_seconds(start, 'full'))
#            last_start = start
            all_characters_list.append(list(temporay_character))
            temporay_character = set()
            all_locations_list.append(list(temporay_location))
            temporay_location = set()
            scene_id = scene
        if i == len(result_sheet)-1:
            scene_time_list.append(time_to_seconds(end,'full'))
#            last_start = start
            all_characters_list.append(list(temporay_character))
            temporay_character = set()
            all_locations_list.append(list(temporay_location))
            temporay_location = set()
            scene_id = scene
        for character in characters.split(","):
            temporay_character.add(character)
        temporay_location.add(location)
          
    return full_length, scene_id, scene_time_list, all_characters_list, all_locations_list

def find_scenes(time_points, video_clips):
    # 创建场景列表，场景的开始和结束时间对
    scenes = [(time_points[i], time_points[i+1]) for i in range(len(time_points) - 1)]
    # 准备结果列表
    result = set()
    for clip in video_clips:
        start = time_to_seconds(clip[0])
        end = time_to_seconds(clip[1])
        if end > time_points[-1]:
            end = time_points[-1]
        # 使用二分查找确定视频片段的开始时间和结束时间分别在哪些场景中
        start_idx = bisect_right(time_points, start) - 1
        end_idx = bisect_left(time_points, end)
        # 将所有相关场景的下标添加到结果集中
        for i in range(start_idx, end_idx):
            result.add(i)
    
    return sorted(result)

def list_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return list(intersection)


def cal_save_factors(result_sheet, question_csv, save_csv_path):
    for i in range(len(question_csv)):
#        print(f"############### {i} #######################")
        script_name = question_csv.iloc[i, 0]
        qType = question_csv.iloc[i, 5]
        question_start_end_list = re_arrange_time(eval(question_csv.iloc[i, 7]))
        
        related_character_list = eval(question_csv.iloc[i, 8])
        related_location_list = eval(question_csv.iloc[i, 9])

        useful_length = Useful_length(question_start_end_list)
        E_time = Entropy_time(question_start_end_list, full_length)
        E_useful_time = cal_usefultime_entropy(question_start_end_list, full_length, time_type="MinSeconds")
        time_factors = [
                           full_length,      # 一集的视频长度(不是输入视频)
                           useful_length,    # 有效片段的长度求和,
                           E_time,           # 时间熵，对于整个视频划分之后的混乱程度
                           E_useful_time,    # 有效时间的熵，对应有效片段所在场景划分之后混乱程度
                           full_length/scene_num   # 场景切换频率
                       ] # to do： shot freq
        
        instance_entropy = cal_instance_entropy(result_sheet, question_start_end_list, related_character_list)
 
        instance_interactions = cal_instance_interactions(result_sheet, question_start_end_list, related_character_list, related_location_list)
        content_factors = [
                              len(qType),    # 相关的topic数量
                              len(all_characters_list)+ len(all_locations_list),       # 所有的人物何地点数量
                              len(related_character_list) + len(related_location_list),  # 相关的人物何地点数量
                              instance_entropy,   # 实例（人物的出现）划分了视频的片段，带来的熵
                              instance_interactions  # 实例可能的交互/关系数量（=实例数量的平方）
                          ]
        data = [[
                   question_csv.iloc[i,0],
                   question_csv.iloc[i,1],
                   question_csv.iloc[i,2],
                   question_csv.iloc[i,3],
                   question_csv.iloc[i,4],
                   question_csv.iloc[i,5], 
                   question_csv.iloc[i,6],
                   question_csv.iloc[i,7],
                   question_csv.iloc[i,8],
                   question_csv.iloc[i,9],
                   time_factors,
                   content_factors
               ]]
        append_to_csv(save_csv_path, data, 'question_complexity')


# 定义正则表达式
pattern = r'\b(Scene|scene)\s+(\d+)'

def parse_question(question):
    scene_id = []
    if 'Scene' in question:
        matches = re.findall(pattern, question)
        for match in matches:
            scene_id.append(int(match[1]))
    return scene_id

def getCorrepondinglines(result_sheet):
    dialogs = dict()
    episode_dialog = ""
    temp_dialog = ""
    key = 1
    for i in range(len(result_sheet)):
        start = result_sheet.iloc[i,1][3:8]
        end = result_sheet.iloc[i,2][3:8]
        dialog = result_sheet.iloc[i,3]
        record_type = result_sheet.iloc[i,5]
        scene_id = result_sheet.iloc[i,6]
        if scene_id != key:
            dialogs[int(key)] = temp_dialog
            temp_dialog = ""
            key = scene_id
        if record_type == 'dialog':
            episode_dialog += f"({start}-{end}): {dialog}\n"
            temp_dialog += f"({start}-{end}): {dialog}\n"
        if i == len(result_sheet) - 1:
            dialogs[int(key)] = temp_dialog
    dialogs['full'] = episode_dialog
    return dialogs



def time_remap(time, initial_time):
    split_time1 = initial_time.split(":")
    split_time2 = time.split(":")
    minute1, second1  = int(split_time1[0]), int(split_time1[1])
    minute2, second2  = int(split_time2[0]), int(split_time2[1])
    minute = minute1 + minute2 + (second1+second2)//60
    second = (second1 + second2)%60
    re_time = "{0:>02d}:{1:02d}".format(minute, second)
    return re_time
    
def getCorrepondinglines_cross_video(result_sheets):
    dialogs = dict()
    key = 1
    episode_dialog_remap = ""
    initial_time = "00:00"
    Up_to_now_time = initial_time
    for order, result_sheet in enumerate(result_sheets):
        episode_dialog = ""
        for i in range(len(result_sheet)):
            start = ":".join(result_sheet.iloc[i,1].strip()[:-4].split(":")[1:])
            end = ":".join(result_sheet.iloc[i,2].strip()[:-4].split(":")[1:])
            dialog = result_sheet.iloc[i,3]
            record_type = result_sheet.iloc[i,5]
            scene_id = result_sheet.iloc[i,6]
            if record_type == 'dialog':
                episode_dialog += f"({start}-{end}): {dialog}\n"
                start = time_remap(start, initial_time)
                end = time_remap(end, initial_time)  
                Up_to_now_time = end
                episode_dialog_remap += f"({start}-{end}): {dialog}\n"
        dialogs[order] = episode_dialog # 单集的，还有重映射的台词都有
        initial_time = Up_to_now_time
    dialogs['full'] = episode_dialog_remap
    return dialogs

def getResult_Sheets(vids):
    result_sheets = []
    for vid in vids:
        if os.path.exists(os.path.join(friends_dir, "result_"+vid+".xlsx")):
            script_path = os.path.join(friends_dir, "result_"+vid+".xlsx")
        else:
            script_path = os.path.join(friends_dir, "result_"+vid+".xltx")
        script = pd.read_excel(script_path,sheet_name=None)
        names = list(script.keys())
        for i, name in enumerate(names):
            if "script" in name:
                 script_id = i 
            if "result" in name:
                 result_id = i
#        print(script_id, result_id)
        result_name = list(script.keys())[result_id]
        result_sheet = script[result_name]
        result_sheets.append(result_sheet)
    return result_sheets
    
def parse_check(file_path):
    check_list = []
    if not os.path.exists(file_path):
        return check_list
    file_name = file_path.split("/")[-1]
#    if int(file_name[8:10])>=2:
#        print("第二季以后暂未检查", end=",")
#        return []
    with open(file_path, 'r') as f:
        eval_str = f.read().strip()
        eval_list = eval_str.split("\n")
        for line in eval_list:
            line_split = line.split(" ")
            check_list.append(line_split)
    return check_list

def get_answer_option(answer_list, GT):
    for i, ans in enumerate(answer_list):
        if ans == GT:
            return chr(65+i)  # 65表示字符'A'
    return None
    
save_dir = "csv"
single_episode_questions = []
cross_episode_questions = []
all_dialogs = dict()
Nomatch = 0
for filename in sorted(os.listdir("csv"))[0:]:
    if not '.csv' in filename:
        continue
    vid = filename.split(".")[0]
    save_csv_path = os.path.join(save_dir, filename)
    if os.path.exists(os.path.join(friends_dir, vid+".xlsx")):
        script_path = os.path.join(friends_dir, vid+".xlsx")
    else:
        script_path = os.path.join(friends_dir, vid+".xltx")
    script = pd.read_excel(script_path,sheet_name=None)
    names = list(script.keys())
    for i, name in enumerate(names):
        if "script" in name:
             script_id = i 
        if "result" in name:
             result_id = i
    print(script_id, result_id)
    result_name = list(script.keys())[result_id]
    result_sheet = script[result_name]
    question_csv = pd.read_csv(save_csv_path)
    question_csv = question_csv.drop_duplicates('question').reset_index(drop=True)
    question_csv.to_csv(save_csv_path, header=True, index=False)
    
    question_csv = pd.read_csv(save_csv_path)
    video_name = question_csv.loc[0,'vid'].split('_')[1]
    all_dialogs[video_name] = getCorrepondinglines(result_sheet)
    claude_check_list = parse_check(f'check/claude/{vid}.txt')
    gemini_check_list = parse_check(f'check/gemini/{vid}.txt')
    print(video_name)
    if len(claude_check_list) == 0 and gemini_check_list ==0:
        print("Not check")
    else:
        if len(claude_check_list) != len(gemini_check_list) or len(claude_check_list) != len(question_csv) or len(gemini_check_list) != len(question_csv) :
            print('fatal error')
            print(len(claude_check_list),len(gemini_check_list),len(question_csv))

    for i in range(len(question_csv)):
        question = question_csv.loc[i,'question']
        choices = eval(question_csv.loc[i,'choices_list'])
        if len(claude_check_list) == 0 or len(gemini_check_list) ==0:
            check_g = ["to_do", "to_do"]
            check_c = ["to_do", "to_do"]
        else:
            check_g = [gemini_check_list[i][1], " ".join(gemini_check_list[i][2:])]
            check_c = [claude_check_list[i][1], " ".join(claude_check_list[i][2:])]
        scene_id = parse_question(question)
        for choice in choices:
            choice_scene_id = parse_question(choice)
            scene_id.extend(choice_scene_id)
        scene_id = list(set(scene_id))
        video_name = question_csv.loc[i,'vid'].split('_')[1]
        answer_list = eval(question_csv.loc[i,'choices_list'])
        GT = question_csv.loc[i,'gt']
        option = get_answer_option(answer_list, GT)
        if not option:
#            print(f'Does not match, {answer_list}, {GT}')
            Nomatch+=1
            continue  # 答案不在选项里面，去掉这个题目
        if len(scene_id) == 0:
           # episode question or cross episodes question
           q_dict = {
               'id': len(single_episode_questions)+1,
               'vid': video_name,
               'scenes': [],
               'question': question, 
               'choices': eval(question_csv.loc[i,'choices_list']),
               'GT': question_csv.loc[i,'gt'],
               'option': option,
               'topic': question_csv.loc[i,'topic'],
               'gemini_check': check_g,
               'claude_check': check_c,
               'attribution': question_csv.loc[i,'attribution'],
               'characters': eval(question_csv.loc[i,'related_person']),	
               'locations': eval(question_csv.loc[i,'related_location']),
               'times': eval(question_csv.loc[i,'related_times']), 
           }
           single_episode_questions.append(q_dict)
        else:
           q_dict = {
               'id': len(single_episode_questions)+1,
               'vid': video_name,
               'scenes': scene_id,
               'question': question, 
               'choices': eval(question_csv.loc[i,'choices_list']),
               'GT': question_csv.loc[i,'gt'],
               'option': option,
               'topic': question_csv.loc[i,'topic'],
               'gemini_check': check_g,
               'claude_check': check_c,
               'attribution': question_csv.loc[i,'attribution'],
               'characters': eval(question_csv.loc[i,'related_person']),	
               'locations': eval(question_csv.loc[i,'related_location']),
               'times': eval(question_csv.loc[i,'related_times']), 
           }
           single_episode_questions.append(q_dict)
print("Delete not match questions：", Nomatch)
# 将列表写入到json文件中
with open('json/single_episode_questions.json', 'w') as f:
    json.dump(single_episode_questions, f, indent=4)
with open('json/dialogs.json', 'w') as f:
    json.dump(all_dialogs, f, indent=4)
    

def split_list(input_list, n):
    output_list = []
    for i in range(0, len(input_list), n):
        output_list.append(input_list[i:i+n])
    return output_list


friends_dir = "scripts/"
save_dir = "csv_cross"
cross_episode_questions = []
all_dialogs = dict()
scripts_name_list = []
for i, name in enumerate(sorted(os.listdir(friends_dir))):
    if 'ori' in name:
        continue
    scripts_name_list.append(os.path.join(friends_dir, name))      # (name) 
split_list = split_list(scripts_name_list, 4)

for i, group in enumerate(split_list[:]):
    print(i, group)
    vids = [path.split("/")[-1][7:-5] for path in group]
    vid_name = vids[0] + "-" +  vids[-1]
    save_csv_path = os.path.join(save_dir, vid_name+".csv")
    if not os.path.exists(save_csv_path):
        continue
    question_csv = pd.read_csv(save_csv_path)
    question_csv = question_csv.drop_duplicates('question').reset_index(drop=True)
    question_csv.to_csv(save_csv_path, header=True, index=False)

    result_sheets = getResult_Sheets(vids)
    all_dialogs[vid_name] = getCorrepondinglines_cross_video(result_sheets)
    print("Now: ",len(question_csv))
    question_csv = pd.read_csv(save_csv_path)
    claude_check_list = parse_check(f'check_cross/claude/{vid_name}.txt')
    gemini_check_list = parse_check(f'check_cross/gemini/{vid_name}.txt')
    if len(claude_check_list) == 0 and gemini_check_list ==0:
        print("Not check")
    else:
        if len(claude_check_list) != len(gemini_check_list) or len(claude_check_list) != len(question_csv) or len(gemini_check_list) != len(question_csv) :
            print('fatal error')
            print(len(claude_check_list),len(gemini_check_list),len(question_csv) )
    for i in range(len(question_csv)):
        question = question_csv.loc[i,'question']
        choices = eval(question_csv.loc[i,'choices_list'])
        if len(claude_check_list) == 0 or len(gemini_check_list) ==0:
            check_g = ["to_do", "to_do"]
            check_c = ["to_do", "to_do"]
        else:
            check_g = [gemini_check_list[i][1], " ".join(gemini_check_list[i][2:])]
            check_c = [claude_check_list[i][1], " ".join(claude_check_list[i][2:])]
     
        answer_list = eval(question_csv.loc[i,'choices_list'])
        GT = question_csv.loc[i,'gt']
        option = get_answer_option(answer_list, GT)
        if not option:
            Nomatch+=1
            continue  # 答案不在选项里面，去掉这个题目
        q_dict = {
           'id': len(cross_episode_questions)+1,
           'vid': vids,
           'question': question, 
           'choices': eval(question_csv.loc[i,'choices_list']),
           'GT': question_csv.loc[i,'gt'],
           'option': option,
           'topic': question_csv.loc[i,'topic'],
           'gemini_check': check_g,
           'claude_check': check_c,
           'attribution': question_csv.loc[i,'attribution'],
           'characters': eval(question_csv.loc[i,'related_person']),	
           'locations': eval(question_csv.loc[i,'related_location']),
           'times': eval(question_csv.loc[i,'related_times']), 
        }
        cross_episode_questions.append(q_dict)
# 将列表写入到json文件中
with open('json/cross_episode_questions.json', 'w') as f:
    json.dump(cross_episode_questions, f, indent=4)
with open('json/cross_dialogs.json', 'w') as f:
    json.dump(all_dialogs, f, indent=4)