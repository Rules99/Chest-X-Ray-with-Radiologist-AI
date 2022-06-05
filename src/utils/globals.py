"""
Feature Selection 

Utils used as global variabeles 

"""
import streamlit as st
import googletrans
# from src.pages import data_analysis,filter_selector,wrapper_selector # import your pages here
from src.pages import bin_class,mlt_class,rassistant,rep_gen
# TO RUN THE APP PARAMETERS
title = 'Radiology Assistant'
navstack = ["Pneumonia Detection",'Pneumonia MultiClassification','Medical Report Generation','Radiology Assistant']
icons = ['bi-diagram-2','bi-diagram-3','bi-body-text','bi-postcard-heart-fill']
funs = [bin_class.app,mlt_class.app,rep_gen.app,rassistant.app]



# Translator variables 
translator = googletrans.Translator(service_urls=['translate.google.com'])
languages = googletrans.LANGUAGES
languages['None'] = 'None'

# Title and description main
title = "Radiology Assistant"
description = '''This is a pocket application that is mainly focused on aiding medical 
    professionals on their diagnostics and treatments for chest anomalies based on chest X-Rays. On this application, users 
    can upload a chest X-Ray image and deep learning models will diagnose anomalies of the image'''


