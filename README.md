# All-Data-Inclusive-Deep-Learning-Models-to-Predict-Critical-Events-in-MIMIC-III
Code For Mimic Dataset Analysis. 
\*Data Processing in this project required a max configration of 8-core 64GB machine.
\*Model Training was performed on K80 Tesla GPU on AI Platform on (Google Cloud Platform)
\*If using GCP we recommend leveraging AI Platform for training as Bayesian Optimization is built into the platform and mutiple models can be trained simultaneously. 
## Dataset Folder Structure
- datasets
   - raw - *This folder contains raw csv files for all data categories except chartevents*
   - chartevents
      - ch_events_first_24_hours_ICUSTAY - *This folder contains chunks of chartevents. Upon exporting the first 24hrs from big query the data is exported in chunks. If you are not using Big Query they kindly split the first 24 hrs. of data into chunks for low memory consumtion.*
      
\* Sub-folders in the dataset-folder are generated as the scripts are executed.
## Code Folder Structure
- All-Data-Inclusive-Deep-Learning-Models-to-Predict-Critical-Events-in-MIMIC-III
   - Data Pre-Processing and Tokenization
   - Data Time Slice Extraction
   - Prepare Training Data
   - Create Models
      - remote_training
         - trainer
   
## Ouput Folder Structure
- output
   - image
   - logs
   - model
   - tokenizer

### Data Pre-Processing and Tokenization
   - Contains notebooks to Preprocess and Tokenize all data categories. Each notebooks name indictaes the data category it process and tokenizes.
### Data Time Slice Extraction
   - Contains notebooks to that extract time slice to relevant data from all data categories data categories.
### Prepare Training Data
   1. **reshape_chartevents_grouped_HADM_ID.py** - group chartevents by HADM_ID.
   2. **reshape_training_all_events_grouped_HADM_ID.py** - group all events except chartevents by HADM_ID.
   3. **Merge_all_events.ipynb** - create all_events.json by HADM_ID and timesteps.
   4. **Create training, valid and test.ipynb** - creates train, valid, and test split. Trains tokenizers, creates vocabulary and integer ecncodes the datasets. 
   5. **Create tfrecords for training, valid and test.ipynb** - Notebook version of script create_tfrecords_train_test_valid.py 
      - or **create_tfrecords_train_test_valid.py - create tfrecords for train, tets and valid**
### Create Model
   - **remote_training**
      - **Remote Training in ml-engine.ipynb** - Contains code to execute remote training on GCP AI-Platform.
      - **hptuning_config.yaml** - Config file containg Hyperparameters for training.
      - **trainer** - This folder contains scripts for remote training. 
   - **Model Evaluvation.ipynb** - To evaluvate model performnace and create AUC-ROC, PR-AUC, Calibration Curve
      - Trained models with least validation error are downloaded from cloud storage to local for Evaluvation. 

