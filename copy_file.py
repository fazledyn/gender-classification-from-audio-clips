import numpy as np
import pandas as pd
import shutil
import os
import csv

IN_FILE  = "./csv/cv-dev-cleaned.csv" 
OUT_FILE = "./cv-dev-cleaned-merge.csv"
OUT_FOLDER = "./cv-dev/"


# rename and copy file
def rename_file(source, destination): 
    filenames = []
    for file in source:
        # print(file)
        split_file = file.split('/')
        # print(split_file)
        new_split_file = split_file[0].split('-')
        # print(new_split_file)
        new_file = new_split_file[1] + '_' + new_split_file[2] + '_' + split_file[1]
        print(new_file)
        os.rename(file, destination + new_file)   
        filenames.append(new_file)
    
    return filenames

def make_csv(file_list, gender_list):
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        # Write the header row
        writer.writerow(["filename", "gender"])

        # Write data in a loop
        for i in range(len(file_list)):
            row = [file_list[i], gender_list[i]]
            writer.writerow(row)

    

if __name__ == '__main__':

    try:
        os.mkdir(OUT_FOLDER)
        os.mkdir(OUT_FOLDER+'/data/')
    except:
        pass

    df = pd.read_csv(IN_FILE)

    filename = df['filename']
    gender = df['gender']
    print(len(filename))

    file_list = rename_file(filename,OUT_FOLDER+'data/')
    print(len(file_list))
    make_csv(file_list,gender)

