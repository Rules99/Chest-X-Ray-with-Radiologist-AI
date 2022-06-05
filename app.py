import os
from src.utils.fun import markdown_
import streamlit as st
# import numpy as np
from PIL import  Image
# Custom imports 
from src.utils.run import  runApp_wt_input
from src.utils.globals import *
from src.utils.traductor.translation import translate_




#
st.set_page_config(layout = 'wide')

# Selecciona un lenguaje

markdown_('WELCOME TO YOUR RADIOLOGY ASSISTANT ! ',tag='h1',style='center')
input = st.selectbox('Select a language:',languages.keys(),format_func = lambda x: languages[x].capitalize().strip(),index = list(languages.keys()).index('None'))
print('INput: ', input)
if input!='None':
  
    
    #### TRANSLATE TITLE AND DESCRIPTION
    title = translate_(translator,title,dest=input)
    description = translate_(translator,description,dest=input)


    

    display = Image.open('./docs/doctors.jpg')
    col1, col2 = st.columns(2)
    col1.image(display, width = 400)
    col2.title(f"{title}")
    col2.write(f'''{description}''')
    # Title of the main page
   
    ## Navigation Option
    # dicnavs = {nav:fun for (nav,fun) in zip(navstack,funs)}
    # Define Dic Navigation 
    navstack = [translate_(translator,nav,dest=input) for nav in navstack]

    runApp_wt_input(title,navstack,icons,funs,styles=None,orientation='horizontal',translator=translator,input=input)


# app = ToRun(dicnavs)
# # Create an instance of the app 
# menu = MultiPageMenu(title,navstack,icons,funs)
# selected = menu.retselected()
# # Run the application
# app.run(selected)











# DECORATOR FUNCTION
# def decMenu(title:str,navstack:list[str],icons:list[str],funs:list[object],
#             styles:dict = None,orientation:str = None):
#            
#                     # Declare temporal dictionary
#                     dicnavs = {nav:fun for (nav,fun) in zip(navstack,funs)}
#                     # Instance menu and class where menu is going to be run
#                     # Define Dic Navigation
#                     app = ToRun(dicnavs)
#                     # Create an instance of the app 
#                     menu = MultiPageMenu(title,navstack,icons,funs)
#                     # Return selected page
#                     selected = menu.retselected()
#                     # Run the main function
#                     fun(*args,**kwargs)
#                     # Run the Application
#                     app.run(selected)