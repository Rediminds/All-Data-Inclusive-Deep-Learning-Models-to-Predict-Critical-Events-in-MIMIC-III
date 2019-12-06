import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import os

###Create training data for all files except chartvents
#creating data for 24 hrs time slice
# create output path
mypath_input = "/home/jupyter/datasets/data_before_24hrs_icu/"
mypath_output = "/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/"
os.makedirs(mypath_output, exist_ok=True)

file_list = list(filter(lambda k: '.json' in k, os.listdir(mypath_input)))

file_list

### Creating a dataset grouped at admission level

# summarize data at HADM_ID and export the training data
for i in tqdm(file_list):
    df = pd.read_json(mypath_input+i, orient='records')
    #retrive column name with 'event' in it
    for j in df.columns:
        if 'event' in j:
            x = j
    df = df.rename({x:'events'}, axis = 'columns')        
    df = df.groupby(by = ['SUBJECT_ID','HADM_ID'], as_index=False).agg({'events':'sum'})
    df.to_json(mypath_output+i)
    
#creating data for 48 hrs time slice
# create output path
mypath_input = "/home/jupyter/datasets/data_before_48hrs_icu/"
mypath_output = "/home/jupyter/datasets/training_data/data_before_48hrs_icu/data_grouped_HADM_ID/"
os.makedirs(mypath_output, exist_ok=True)

file_list = list(filter(lambda k: '.json' in k, os.listdir(mypath_input)))

file_list

### Creating a dataset grouped at admission level

# summarize data at HADM_ID and export the training data
for i in tqdm(file_list):
    df = pd.read_json(mypath_input+i, orient='records')
    #retrive column name with 'event' in it
    for j in df.columns:
        if 'event' in j:
            x = j
    df = df.rename({x:'events'}, axis = 'columns')        
    df = df.groupby(by = ['SUBJECT_ID','HADM_ID'], as_index=False).agg({'events':'sum'})
    df.to_json(mypath_output+i)