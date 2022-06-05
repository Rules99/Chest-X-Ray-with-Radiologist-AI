
from src.utils.multipage import MultiPageMenu,ToRun
import streamlit as st
import pandas as pd
# Navigation Option
# RUN FUNCTION
def runApp(title:str,navstack:list,icons:list,funs:list,
            styles:dict = None,orientation:str = 'horizontal'):
    '''
    Run the application intended
    @ param title (str): title of the application
    @ param navstack (list[str]): navigation string options
    @ param icons (list[str]): icons of each navstack element
    @ param funs (list[function]) : each of the running scripts 
    @ param styles (dict): style of the feature selection
    @ param orientation (str) : orientation of the navigation list 
    '''      
    # Declare temporal dictionary
    dicnavs = {nav:fun for (nav,fun) in zip(navstack,funs)}
    # Instance menu and class where menu is going to be run
    # Define Dic Navigation
    app = ToRun(dicnavs)
    # Create an instance of the app 
    menu = MultiPageMenu(title,navstack,icons,funs,styles = styles, orientation=orientation)
    # Return selected page
    selected = menu.retselected()
    # Run the Application
    app.run(selected)
# RUN FUNCTION WITH DATA INPUTED
def runAppdata(data:pd.DataFrame,title:str,navstack:list,icons:list,funs:list,
            styles:dict = None,orientation:str = 'horizontal'):
    '''
    Run the application intended
    @ param dataframe(pd.Dataframe) :  datafrane of the feature selector
    @ param title (str): title of the application
    @ param navstack (list[str]): navigation string options
    @ param icons (list[str]): icons of each navstack element
    @ param funs (list[function]) : each of the running scripts 
    @ param styles (dict): style of the feature selection
    @ param orientation (str) : orientation of the navigation list 
    '''      
    # Declare temporal dictionary
    dicnavs = {nav:fun for (nav,fun) in zip(navstack,funs)}
    # Instance menu and class where menu is going to be run
    # Define Dic Navigation
    app = ToRun(dicnavs)
    # Create an instance of the app 
    menu = MultiPageMenu(title,navstack,icons,funs,styles = styles, orientation=orientation)
    # Return selected page
    selected = menu.retselected()
    # Run the Application
    app.run_data(selected,data)


def runApp_wt_input(title:str,navstack:list,icons:list,funs:list,
            styles:dict = None,orientation:str = 'horizontal',translator:object = None,input:str = 'es'):
    '''
    Run the application intended
    @ param title (str): title of the application
    @ param navstack (list[str]): navigation string options
    @ param icons (list[str]): icons of each navstack element
    @ param funs (list[function]) : each of the running scripts 
    @ param styles (dict): style of the feature selection
    @ param orientation (str) : orientation of the navigation list 
    '''      
    # Declare temporal dictionary
    dicnavs = {nav:fun for (nav,fun) in zip(navstack,funs)}
    # Instance menu and class where menu is going to be run
    # Define Dic Navigation
    app = ToRun(dicnavs)
    # Create an instance of the app 
    menu = MultiPageMenu(title,navstack,icons,funs,styles = styles, orientation=orientation)
    # Return selected page
    selected = menu.retselected()
    # Run the Application
    app.run_wt_input(selected,translator,input)

# def runApplication(allow_input_data:bool = False,dataframe:pd.DataFrame = None):
#         if not allow_input_data:
#                 dataframe = None
#         def decorateApp(fun):
#                 def App(title:str,navstack:list[str],icons:list[str],funs:list[object],
#                         styles:dict = None,orientation:str = 'horizontal'):

#                         '''
#                         Run the application intended
#                         @ param title (str): title of the application
#                         @ param navstack (list[str]): navigation string options
#                         @ param icons (list[str]): icons of each navstack element
#                         @ param funs (list[function]) : each of the running scripts 
#                         @ param styles (dict): style of the feature selection
#                         @ param orientation (str) : orientation of the navigation list 
#                         '''      
#                         selected,app = fun(title,navstack,icons,funs,styles,orientation)
#                         # Run the Application
#                         if allow_input_data:
#                                 app.run_data(selected,dataframe)
#                         else:
#                                 app.run(selected)
#                         return App
#                 return decorateApp

