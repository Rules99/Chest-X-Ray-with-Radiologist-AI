
# from email.encoders import encode_noop
from src.features.CNN_encoder import CNN_Encoder
from src.data.configs_nlp import argHandler
from src.features.tokenizer_wrapper import TokenizerWrapper
from src.models.gpt2_model import TFGPT2LMHeadModel

# Nuestras
from src.utils.fun import markdown_
from src.utils.traductor.translation import translate_
from src.utils.traductor.strings_diagnostic import *
from src.utils.traductor.general_strings import *



import streamlit as st
# from io import StringIO
from skimage.transform import resize
import tensorflow as tf
# import os
import numpy as np
from PIL import Image
# import json
import time
# import pandas as pd
# from tqdm import tqdm



def preprocess_gen_rep(image_file:str):
    """
    Preprocesar las imagenes para la generacion del report

    Args:
        image_file (str): Archivo ruta
        directory (str): directorio ruta

    Returns:
        _type_: _description_
    """    
    # Load the image

    image = Image.open(image_file)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (224, 224, 3))

    # Normalize
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - imagenet_mean) / imagenet_std
    return image_array[None,:,:,:]




def evaluate_full(FLAGS, encoder, decoder, tokenizer_wrapper, images):
    avg_time = 0
    step_n = 1
    visual_features, tags_embeddings = encoder(images)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    # dec_input = tf.tile(dec_input,[images.shape[0],1])
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])
    start_time = time.time()
    
    tokens = decoder.generate(dec_input, max_length=FLAGS.max_sequence_length, num_beams=num_beams, min_length=3,
                              eos_token_ids=[tokenizer_wrapper.GPT2_eos_token_id()], no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)
    end_time = time.time() - start_time
    # print(f"Step time: {end_time}")
    avg_time += end_time
    # print(f"avg Step time: {avg_time / step_n}")

    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)


    print(f'''
        Getting the sentece {step_n}  : {sentence}


          ''')
    step_n += 1
    return sentence

def predict_report(image, FLAGS, encoder, decoder, tokenizer_wrapper)->str:
    """
    Predecir el informe diagnostico

    Args:
        image (_type_): _description_
        FLAGS (_type_): _description_
        encoder (_type_): _description_
        decoder (_type_): _description_
        tokenizer_wrapper (_type_): _description_

    Returns:
        _type_: _description_
    """    
    tf.keras.backend.set_learning_phase(0)
    # images, target, img_path = next(generator)
    
    
    predicted_sentence = evaluate_full(FLAGS, encoder, decoder, tokenizer_wrapper,
                                    image)
    print('Predicted sentence: ', predicted_sentence)
    return predicted_sentence  


def pred_nlp_model(img_path:str)->str:
    FLAGS = argHandler()
    FLAGS.setDefaults()
    

    tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                        FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

   

    image = preprocess_gen_rep(img_path)
    # test_enqueuer, test_steps = get_enqueuer(FLAGS.test_csv, 1, FLAGS, tokenizer_wrapper)
    # test_enqueuer.start(workers=1, max_queue_size=8)

    path = 'models/checkpoints/pretrained_visual_model'
    encoder = CNN_Encoder(path, FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                        FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)
    # print('''Latest Checkpoint ''' ,
    #     ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
    # evaluate_enqueuer(test_enqueuer, test_steps, FLAGS, encoder, decoder, tokenizer_wrapper, write_images=True, test_mode=True)
    return predict_report(image, FLAGS, encoder, decoder, tokenizer_wrapper)



def postprocess(prediction_text:str,translator:object,input:str)->str:
    """
    Post procesamiento de la predicción para que se entienda mejor

    Args:
        prediction_text (str): _description_
        translator (object): _description_
        input (str): _description_

    Returns:
        str: _description_
    """    
    # # Procesamiento de tecto
    prediction_text = prediction_text.replace('"','')
    for i in range(0,10):
        try:
            prediction_text = prediction_text.replace(str(i)+'.','')
        except:
            try:
                prediction_text = prediction_text.replace(str(i),'')
            except:
                pass
            pass

    string = str()
    for i in list(  set(    [text.strip() for text in prediction_text.split('.')]   )   ):
        try:
            temp = translate_(translator,i,dest = input)
            string+=f"{temp.capitalize()}"+'.'
            string+="<br>"
        except TypeError:
            print('Error')
    return string
                        
def gen_report(uploaded_file:str,translator:object,input:str):
    """
    Funcion para generar el report

    Args:
        uploaded_file (object): archivo a subir
        translator (object, optional): objeto para traducir los strings. Defaults to None.
        input (str, optional): destino de traduccion. Defaults to 'en'.
    """    
    __,col1,__,col2,__ = st.columns([0.2,1,0.3,1,0.2])
    # Original image
    with col1:
        markdown_(translate_(translator,or_im_output,dest=input),'h2',style='center')
        st.image(Image.open(uploaded_file),use_column_width=True)
    # Prediction Image
    with col2:
    
        # Predicción
        prediction_text = pred_nlp_model(uploaded_file)
        # Post proceso
        prediction_text = postprocess(prediction_text,translator,input)

        markdown_(translate_(translator,rep_output,dest=input),'h2',style='center')
        markdown_(prediction_text,'p',style = 'center')


def app(translator:object,input:str):
     
    uploaded_file = st.file_uploader("Choose an X-Ray image to detect anomalies of the chest (the file must be a dicom extension or jpg)")

    if uploaded_file is not None:
       gen_report(uploaded_file,translator,input)
