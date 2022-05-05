import os
import shutil

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')


curr_dir = os.getcwd()
models_dir = curr_dir + "/models/"
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42


bert_model_name = 'bert_en_cased_L-12_H-768_A-12' 


tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

@st.cache
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)


#Hypo Red Dataset

#Training
classifier_model = build_classifier_model()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 3 # 1 epoch for hypo red and 2 for hypo dataset. After that, the model overfits.
#steps_per_epoch = len(x_train)//batch_size
steps_per_epoch = 3785//batch_size
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
#print(num_train_steps)
#print(num_warmup_steps)
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

@st.cache
reloaded_model = tf.keras.models.load_model(models_dir + 'hypo_red_trained_bert_cased_e3.h5',  custom_objects = {'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer})
#test_loss, test_acc = reloaded_model.evaluate(x_test, y_test)

#print('Test accuracy:', test_acc)

def inference(examples):
  #examples = ['I know programming like the back of my palm', 'I know programming very well']

  results = tf.sigmoid(reloaded_model(tf.constant(examples)))
  return results

