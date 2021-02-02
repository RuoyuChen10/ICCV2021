# -*- coding: utf-8 -*-  

"""
Created on 2021/1/30

@author: Ruoyu Chen
"""

import os
import pandas as pd
import random

from tqdm import tqdm

# Dir of the MAAD-Face label csv
MAAD_Face_dir = "/home/cry/data1/ICCV2021/Datasets/MAAD_Face.csv"
datasets_label = "./datasets.txt"

# remove files
def remove(path):
    if os.path.exists(path):
        os.remove(path)

# split the datasets
def train_test_split(datasets_information):
    # distribute
    random.shuffle(datasets_information)
    # remove exisit label txt
    remove("./training.txt");remove("./val.txt");remove("./test.txt")
    num = len(datasets_information)
    # write
    for i in tqdm(range(0,int(num*0.7))):
        with open("./training.txt","a") as file:
            file.write(datasets_information[i])
    for i in tqdm(range(int(num*0.7),int(num*0.7)+int(num*0.15))):
        with open("./val.txt","a") as file:
            file.write(datasets_information[i])
    for i in tqdm(range(int(num*0.7)+int(num*0.15),num)):
        with open("./test.txt","a") as file:
            file.write(datasets_information[i])

def main():
    # names = ['Filename','Male','Young','Middle_Aged','Senior','Asian','White','Black']
    datasets = pd.read_csv(MAAD_Face_dir,usecols=[0,2,3,4,5,6,7,8])

    datasets_information = []
    for i in tqdm(range(len(datasets))):
        # Gender:
        if datasets['Male'][i] == 1:
            idx_1 = 0
        elif datasets['Male'][i] == -1:
            idx_1 = 1
        else:
            continue
        # Age:
        if datasets['Young'][i] == 1:
            idx_2 = 0
        elif datasets['Middle_Aged'][i] == 1:
            idx_2 = 1
        elif datasets['Senior'][i] == 1:
            idx_2 = 2
        else:
            continue
        # Race:
        if datasets['Asian'][i] == 1:
            idx_3 = 0
        elif datasets['White'][i] == 1:
            idx_3 = 1
        elif datasets['Black'][i] == 1:
            idx_3 = 2
        else:
            continue
        information = datasets['Filename'][i] + ' ' + str(idx_1) + ' ' + str(idx_2) + ' ' + str(idx_3)+'\n'
        datasets_information.append(information)
    train_test_split(datasets_information)

if __name__ == '__main__':
    main()