import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import os

# create output path
mypath_input = "/home/jupyter/datasets/chartevents/tokenized/ch_events_chunks_ungrouped/"
mypath_output = "/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/"
os.makedirs(mypath_output, exist_ok=True)

chunk_list = list(filter(lambda k: '.json' in k, os.listdir(mypath_input)))

df1 = pd.DataFrame()
for i in tqdm(chunk_list):
    df = pd.read_json(mypath_input+i, orient='records')
    for j in df.columns:
        if 'event' in j:
            x = j
    df = df.rename({x:'events'}, axis = 'columns')   
    df1 = df1.append(df.groupby(by = ['SUBJECT_ID','HADM_ID'], as_index=False).agg({'events':'sum','HOSPITAL_EXPIRE_FLAG':'mean'}),ignore_index = True)   

df1 = df1.groupby(by = ['SUBJECT_ID','HADM_ID'], as_index=False).agg({'events':'sum','HOSPITAL_EXPIRE_FLAG':'mean'})

df1.to_json(mypath_output+'chartevents.json')