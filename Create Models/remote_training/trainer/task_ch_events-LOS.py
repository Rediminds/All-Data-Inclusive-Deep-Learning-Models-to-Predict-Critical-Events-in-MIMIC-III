import pickle
from time import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend
from tensorflow.python.lib.io import file_io
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("--train_input", help="Specify the path to training file", type=str, required=True)
parser.add_argument("--valid_input", help="Specify the path to validation file", type=str, required=True)
parser.add_argument("--test_input", help="Specify the path to test file", type=str, required=True)
parser.add_argument("--output_path", help="Specify the path output folder", type=str, required=True)
parser.add_argument("--num_dense_layers", help="Specify num of hidden layers", type=int, required=True)
parser.add_argument("--neurons", help="Specify num of nuerons in hidden layers", type=int, required=True)
parser.add_argument("--batch_size", help="Specify the bacth size", type=int, required=True)
parser.add_argument("--learning_rate", help="Specify learning rate", type=float, required=True)
parser.add_argument("--num_epochs", help="Specify the num of epochs for training", type=int, required=True)
parser.add_argument("--dropout", help="Specify the ratio of dense dropout", type=float, required=True)
parser.add_argument("--embedding_dropout", help="Specify the ratio of embedding dropout", type=float, required=True)
parser.add_argument("--embed_dim", help="Specify embedding dimension", type=int, required=True)
parser.add_argument("--hypertune", help="specify 'hypertune' if hypertuning parameters else 'nothypertune'", type=str, required=True)

args = parser.parse_args()



# set Path to train, valid, test files
train_input = args.train_input
valid_input = args.valid_input
test_input = args.test_input

#count the number of records in train,test and valid
train_records = sum(1 for _ in tf.python_io.tf_record_iterator(train_input))
valid_records = sum(1 for _ in tf.python_io.tf_record_iterator(valid_input))
test_records = sum(1 for _ in tf.python_io.tf_record_iterator(test_input))

# combine mutiple files and import through tfRecordDataset
train_filenames = [train_input]
valid_filenames = [valid_input]
test_filenames = [test_input]
train_dataset = tf.data.TFRecordDataset(train_filenames)
valid_dataset = tf.data.TFRecordDataset(valid_filenames)
test_dataset = tf.data.TFRecordDataset(test_filenames)

# Create a description of the features for parsing the input files
feature_description = {
    'LOS': tf.FixedLenFeature([], dtype=tf.int64),
    'feature1': tf.VarLenFeature(dtype=tf.int64),
    'feature2': tf.VarLenFeature(dtype=tf.int64),
    'feature3': tf.VarLenFeature(dtype=tf.int64),
    'feature4': tf.VarLenFeature(dtype=tf.int64),
    'feature5': tf.VarLenFeature(dtype=tf.int64),
    'feature6': tf.VarLenFeature(dtype=tf.int64),
    'feature7': tf.VarLenFeature(dtype=tf.int64),
    'feature8': tf.VarLenFeature(dtype=tf.int64),
    'feature9': tf.VarLenFeature(dtype=tf.int64)
}

# Funtion to parse input files
def _parse_function(example_proto):
    x = tf.parse_single_example(example_proto, feature_description)
    label = tf.cast(x['LOS'],dtype='int32')
    ch_events = tf.cast(tf.sparse.to_dense(x['feature1']),dtype='int32')
    inputcv_events = tf.cast(tf.sparse.to_dense(x['feature2']),dtype='int32')
    inputmv_events = tf.cast(tf.sparse.to_dense(x['feature3']),dtype='int32')
    lab_events = tf.cast(tf.sparse.to_dense(x['feature4']),dtype='int32')
    microbio_events = tf.cast(tf.sparse.to_dense(x['feature5']),dtype='int32')
    note_events = tf.cast(tf.sparse.to_dense(x['feature6']),dtype='int32')
    output_events = tf.cast(tf.sparse.to_dense(x['feature7']),dtype='int32')
    prescription_events = tf.cast(tf.sparse.to_dense(x['feature8']),dtype='int32')
    procedure_events = tf.cast(tf.sparse.to_dense(x['feature9']),dtype='int32')
    
    return ((ch_events),
            label)#{'labels':label, 'ch_events':ch_events}


# map the parsing funtion to dataset
train_parsed_dataset = train_dataset.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_parsed_dataset = valid_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_parsed_dataset = test_dataset.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Set bacth size for datset generator and number of repeats per epoch
batch_size = args.batch_size
buffer_size = 5
train_parsed_dataset = train_parsed_dataset.batch(batch_size)
train_parsed_dataset = train_parsed_dataset.repeat()
train_parsed_dataset = train_parsed_dataset.prefetch(buffer_size)

valid_parsed_dataset = valid_parsed_dataset.batch(batch_size)
valid_parsed_dataset = valid_parsed_dataset.repeat()
valid_parsed_dataset = valid_parsed_dataset.prefetch(buffer_size)

#import tokenizer
import json
tokenizer_dict = {'tokenizer_chartevents': 354107,
                  'tokenizer_inputevents_cv': 310344,
                  'tokenizer_inputevents_mv': 2389823,
                  'tokenizer_labevents': 128492,
                  'tokenizer_microbioevents': 806,
                  'tokenizer_noteevents': 1864028,
                  'tokenizer_outputevents': 26261,
                  'tokenizer_prescriptionevents': 70476,
                  'tokenizer_procedureevents': 53103}    #json.load(open(args.tokenizer_path+"tokenizer_dict.json"))

embed_dim = {}
for i in tokenizer_dict:
    embed_dim.update({i:args.embed_dim})


                           
# Define a custom layer for averaging inputs    
def call(inputs, mask=None):
    steps_axis = 1
    if mask is not None:
        mask = math_ops.cast(mask, backend.floatx())
        input_shape = inputs.shape.as_list()
        broadcast_shape = [-1, input_shape[steps_axis], 1]
        mask = array_ops.reshape(mask, broadcast_shape)
        inputs *= mask
        return backend.sum(inputs, axis=steps_axis) / (math_ops.reduce_sum(mask, axis=steps_axis)+backend.epsilon())
    else:
        return backend.mean(inputs, axis=steps_axis)

#Keras Network for training 


# Embedding layer for ch_events
input_ch_event = keras.layers.Input(shape=(11206,), name = "ch_events")
x_1 = keras.layers.Embedding(output_dim=embed_dim['tokenizer_chartevents'],
                             input_dim=tokenizer_dict['tokenizer_chartevents']+2,
                             mask_zero=True,
                             name = 'embedding_layer_ch')(input_ch_event)

#x_1 = keras.layers.GlobalAveragePooling1D(data_format='channels_last')(x_1)


x = keras.layers.SpatialDropout1D(rate = args.embedding_dropout)(x_1)

x = keras.layers.Lambda(lambda x: call(x))(x)

#dense layers for training
for i in range(args.num_dense_layers):
    x = keras.layers.Dense(args.neurons,
                           activation='relu', 
                           name='dense_'+str(i))(x)
    x = keras.layers.Dropout(args.dropout)(x)
    
main_output = keras.layers.Dense(1, activation='sigmoid',name='main_output')(x)

model = keras.Model(inputs=[input_ch_event], outputs=main_output)

# compiling the model
model.compile(optimizer=keras.optimizers.Adam(lr = args.learning_rate), loss= 'binary_crossentropy' , metrics = ['acc'])

# Keras callback. The patience parameter is the amount of epochs to check for improvement
"""early_stop = keras.callbacks.EarlyStopping(monitor= 'val_loss', 
                                           patience=5, 
                                           mode='min',
                                           restore_best_weights=True)"""


#set output_dir for tensorboard logs
if args.hypertune=="hypertune":
    # if tuning, join the trial number to the output path
    trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    output_dir = os.path.join(args.output_path,trial)
else:
    output_dir = os.path.join(args.output_path)

# Setup TensorBoard callback.    
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir,'keras_tensorboard','log_{}'.format(time())))


# Setup Checkpoint callback path
if args.hypertune=="hypertune":
    checkpoint_path = 'best_model.h5'

else:
    checkpoint_dir = os.path.join(output_dir,'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir,'best_model.h5')

# Setup Checkpoint callback.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    


model.summary()

model.fit(train_parsed_dataset,
          epochs=args.num_epochs, 
          steps_per_epoch=int(train_records/batch_size)+1,
          validation_data=valid_parsed_dataset,
          validation_steps=int(valid_records/batch_size)+1,
          callbacks=[tensorboard_cb, checkpoint]
         )

# Export the model to a local SavedModel directory/cloud storage
export_path = tf.contrib.saved_model.save_keras_model(model, saved_model_path=os.path.join(output_dir,"model"))

# export checkpoint to cloud storage output directory
if args.hypertune=="hypertune":
    with file_io.FileIO(checkpoint_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(output_dir,'checkpoint',checkpoint_path), mode='wb+') as output_f:
            output_f.write(input_f.read())


test_parsed_dataset = test_parsed_dataset.batch(batch_size)
test_parsed_dataset = test_parsed_dataset.repeat()
model.evaluate(test_parsed_dataset,steps=int(test_records/batch_size)+1)





