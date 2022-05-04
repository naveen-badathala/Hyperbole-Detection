from importlib import reload
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
#from official.nlp.optimization 
import os
#from eval import evaluate


curr_dir = os.getcwd()
models_dir = curr_dir + "/models/"
st.set_page_config(layout="wide")
st.title("Hyperbole-Detection")

def my_widget(key):
    return st.button("Submit ")

#option = st.selectbox(
#     'Select the type of task',
#     ('Single Sentence', 'Two Sentence'))

#st.write('You selected:', option)


with st.form(key='my_form'):
    
    #print(option)
    
    #if option == 'Single Sentence':

    text_area_input1 = st.text_area("Enter Sentence:")

    #else:
    #    text_input2 = st.text_input("Enter Sentence-1:")
    #    text_input3 = st.text_input("Enter Sentence-2:")


    submitButton = st.form_submit_button(label = 'Predict')




if submitButton:
    #print(text_area_input)
    #print(text_input)
    examples = []
    #if option == 'Single Sentence':
    examples.append(text_area_input1)
   # else:
   #     examples.append(text_input2)
   #     examples.append(text_input3)
    #init_lr = 3e-5
    optimizer = optimization.create_optimizer(optimizer_type='adamw')
    reloaded_model = tf.keras.models.load_model(models_dir + 'hypo_red_trained_bert_cased_e3.h5',  custom_objects = {'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer})
    #reloaded_model = reloaded_model = tf.keras.models.load_model(models_dir + 'hypo_red_trained_bert_cased_e3.h5')
    results = tf.sigmoid(reloaded_model(tf.constant(examples)))
    answer = "HI"
    st.markdown(
       f"""
        * Answer : {results}
        """
    )