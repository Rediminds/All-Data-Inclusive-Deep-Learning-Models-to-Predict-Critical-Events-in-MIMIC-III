#import packages
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm.autonotebook import tqdm

# import data
path_train_dataset = '/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/train_test_valid/all_events_train.json'
path_valid_dataset = '/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/train_test_valid/all_events_valid.json'
path_test_dataset = '/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/train_test_valid/all_events_test.json'

df_train = pd.read_json(path_train_dataset)
df_valid = pd.read_json(path_valid_dataset)
df_test = pd.read_json(path_test_dataset)

# impost dictionary to define padding
dict_path = '/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/padded_arrays/pctl999_padlen_dict.json'

import json

with open(dict_path) as filehandle:
    pctl_999_dict = json.load(filehandle)
mypath_output = "/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/padded_arrays/"

#serialize target and feature to single tf-record
def serialize_example(target,target2,feature1, feature2,feature3,feature4, feature5,feature6,feature7,feature8,feature9):
    """
    Creates a tf.Example message ready to be written to a file.
    """
  
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
    feature = {
      'HOSPITAL_EXPIRE_FLAG': tf.train.Feature(int64_list=tf.train.Int64List(value=[target])),
      'LOS': tf.train.Feature(int64_list=tf.train.Int64List(value=[target2])),
      'feature1': tf.train.Feature(int64_list=tf.train.Int64List(value=feature1)),
      'feature2': tf.train.Feature(int64_list=tf.train.Int64List(value=feature2)),
      'feature3': tf.train.Feature(int64_list=tf.train.Int64List(value=feature3)),
      'feature4': tf.train.Feature(int64_list=tf.train.Int64List(value=feature4)),
      'feature5': tf.train.Feature(int64_list=tf.train.Int64List(value=feature5)),
      'feature6': tf.train.Feature(int64_list=tf.train.Int64List(value=feature6)),
      'feature7': tf.train.Feature(int64_list=tf.train.Int64List(value=feature7)),
      'feature8': tf.train.Feature(int64_list=tf.train.Int64List(value=feature8)),
      'feature9': tf.train.Feature(int64_list=tf.train.Int64List(value=feature9))
    }
  
  # Create a Features message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


### Create tr.record for test data

mypath_output = '/home/jupyter/datasets/training_data/data_before_24hrs_icu/data_grouped_HADM_ID/padded_arrays/'

# Write the `tf.Example` observations to the file.
filename = 'all_events_test.tfrecord'
with tf.python_io.TFRecordWriter(mypath_output+filename) as writer:
    for i in tqdm(range(len(df_test))):
        
        example = serialize_example(target = df_test['HOSPITAL_EXPIRE_FLAG'][i],
                                    target2 = df_test['LOS'][i],
                                    feature1 = pad_sequences([df_test['event'][i]], maxlen=pctl_999_dict['event'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature2 = pad_sequences([df_test['inputevents_cv'][i]], maxlen=pctl_999_dict['inputevents_cv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature3 = pad_sequences([df_test['inputevents_mv'][i]], maxlen=pctl_999_dict['inputevents_mv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature4 = pad_sequences([df_test['labevents'][i]], maxlen=pctl_999_dict['labevents'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature5 = pad_sequences([df_test['microbioevents'][i]], maxlen=pctl_999_dict['microbioevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature6 = pad_sequences([df_test['noteevents'][i]], maxlen=pctl_999_dict['noteevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature7 = pad_sequences([df_test['outputevents'][i]], maxlen=pctl_999_dict['outputevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature8 = pad_sequences([df_test['prescriptionevents'][i]], maxlen=pctl_999_dict['prescriptionevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature9 = pad_sequences([df_test['procedureevents'][i]], maxlen=pctl_999_dict['procedureevents'], dtype='int32', padding='post', truncating='pre', value=0)[0])
        writer.write(example)


# Write the `tf.Example` observations to the file.
filename = 'all_events_valid.tfrecord'
with tf.python_io.TFRecordWriter(mypath_output+filename) as writer:
    for i in tqdm(range(len(df_valid))):
        
        example = serialize_example(target = df_valid['HOSPITAL_EXPIRE_FLAG'][i],
                                    target2 = df_valid['LOS'][i],
                                    feature1 = pad_sequences([df_valid['event'][i]], maxlen=pctl_999_dict['event'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature2 = pad_sequences([df_valid['inputevents_cv'][i]], maxlen=pctl_999_dict['inputevents_cv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature3 = pad_sequences([df_valid['inputevents_mv'][i]], maxlen=pctl_999_dict['inputevents_mv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature4 = pad_sequences([df_valid['labevents'][i]], maxlen=pctl_999_dict['labevents'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature5 = pad_sequences([df_valid['microbioevents'][i]], maxlen=pctl_999_dict['microbioevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature6 = pad_sequences([df_valid['noteevents'][i]], maxlen=pctl_999_dict['noteevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature7 = pad_sequences([df_valid['outputevents'][i]], maxlen=pctl_999_dict['outputevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature8 = pad_sequences([df_valid['prescriptionevents'][i]], maxlen=pctl_999_dict['prescriptionevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature9 = pad_sequences([df_valid['procedureevents'][i]], maxlen=pctl_999_dict['procedureevents'], dtype='int32', padding='post', truncating='pre', value=0)[0])
        writer.write(example)

# Write the `tf.Example` observations to the file.
filename = 'all_events_train.tfrecord'
with tf.python_io.TFRecordWriter(mypath_output+filename) as writer:
    for i in tqdm(range(len(df_train))):
        
        example = serialize_example(target = df_train['HOSPITAL_EXPIRE_FLAG'][i],
                                    target2 = df_train['LOS'][i],
                                    feature1 = pad_sequences([df_train['event'][i]], maxlen=pctl_999_dict['event'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature2 = pad_sequences([df_train['inputevents_cv'][i]], maxlen=pctl_999_dict['inputevents_cv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature3 = pad_sequences([df_train['inputevents_mv'][i]], maxlen=pctl_999_dict['inputevents_mv'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature4 = pad_sequences([df_train['labevents'][i]], maxlen=pctl_999_dict['labevents'], dtype='int32', padding='post', truncating='pre', value=0)[0], 
                                    feature5 = pad_sequences([df_train['microbioevents'][i]], maxlen=pctl_999_dict['microbioevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature6 = pad_sequences([df_train['noteevents'][i]], maxlen=pctl_999_dict['noteevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature7 = pad_sequences([df_train['outputevents'][i]], maxlen=pctl_999_dict['outputevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature8 = pad_sequences([df_train['prescriptionevents'][i]], maxlen=pctl_999_dict['prescriptionevents'], dtype='int32', padding='post', truncating='pre', value=0)[0],
                                    feature9 = pad_sequences([df_train['procedureevents'][i]], maxlen=pctl_999_dict['procedureevents'], dtype='int32', padding='post', truncating='pre', value=0)[0])
        writer.write(example)