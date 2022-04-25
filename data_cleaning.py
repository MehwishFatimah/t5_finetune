#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:55:19 2022
@author: fatimamh

This codes clean the raw Wikipedia files to prepare clean input for t5 finetuning.
Text is without title and headings.
"""

import os
import re
import pandas as pd
import time
import resource
import config as config

class Data(object):
    def __init__(self):

        pass
        
    '''---------------------------------------------'''
    def replace_tags_a(self, text):
         
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        reg = "<TITLE> (.*?) </TITLE>"
        title = re.findall(reg, text)
        title = title[0]
        reg = "<HEADING> (.*?) </HEADING> <SECTION> (.*?) </SECTION>"
        res = re.findall(reg, text)
        
        art = []
        for h, sec in res:
            reg = "<S> (.*?) </S>"
            st = re.findall(reg, sec)
            for s in st:
                art.append(s)
            
        text = ' '.join(art)
        text = text.lstrip()
        
        #print('text:{}\n'.format(text))
        return text

    '''---------------------------------------------'''
    def replace_tags_s(self, text):

        text = text.replace('<SUMMARY>', ' ')
        text = text.replace('</SUMMARY>', ' ')
        text = text.replace('<S>', ' ')
        text = text.replace('</S>', ' ')
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        text = text.lstrip()
        
        #print('summary:{}\n'.format(text))
        return text

    '''---------------------------------------------'''
    def clean_data(self, df):

        print('Data before cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
        
        df['text']    = df['text'].apply(lambda x: self.replace_tags_a(x))
        df['summary'] = df['summary'].apply(lambda x: self.replace_tags_s(x))

        if 'index' in df.columns:
            del df['index']

        print('Data after cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
        return df

    '''---------------------------------------------'''
    def process_data(self, ext = '.csv'):

        files = config.files 
        print(files)
        folder = config.root
        out_folder = config.out_folder

        for file in files:
            file_name = os.path.splitext(file)[0]
            print(file_name)
            file = os.path.join(folder, file)
            df   = pd.read_json(file, encoding = 'utf-8')

            if 'dsummary' in df.columns:
                del df['dtext']
                del df['summary']
                df = df.rename(columns={'dsummary': 'summary'})
            
            #df = df.head(5)
            df   = self.clean_data(df)
            print('\n--------------------------------------------')

            file = os.path.join(out_folder, file_name + ext)
            print(file)
            df.to_csv(file, index = False)
            print('\n======================================================================================')


if __name__ == "__main__":

    start_time = time.time()
    # Step 1: Convert data from json to csv. CLEAN 
    data = Data()
    print("cleaning data")
    data.process_data()
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
    format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))