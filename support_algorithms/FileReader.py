import os
import re
import sys
import cchardet
from collections import defaultdict

# file reader function
def read_files(file_path):
    file_data=defaultdict(list)
    with open(file_path,'rb') as detect_file:
        info=cchardet.detect(detect_file.read())
    with open(file_path,'r+',encoding=info['encoding']) as input_file:
        lines=input_file.readlines()
        for line in lines:
            elements=line.split(' ')
            element_num=len(elements)
            # contain data in this line
            if elements[0][0] is 'E':
                for temp_a in range(element_num):
                    if temp_a==0:
                        file_data['name'].append(elements[temp_a])
                    elif temp_a==element_num-1:
                        file_data['label'].append(re.sub(r'\n','',elements[temp_a]))
                    else:
                        file_data[temp_a].append(float(elements[temp_a]))
    return file_data

# folder_reader function
def read_folder(folder_directory,folder_name):
    folder_path=os.path.join(folder_directory,folder_name)
    file_names=os.listdir(folder_path)
    folder_content={}
    for file_name in file_names:
        file_path=os.path.join(folder_path,file_name)       
        file_data=read_files(file_path)
        folder_content[file_name]=file_data
        # folder_content is a dictionary containing sub-dictionaries
    return folder_content