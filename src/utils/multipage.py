"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st
# import streamlit.components.v1 as html
from streamlit_option_menu import option_menu

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        # Drodown to select the page to run  
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        # run the app function 
        page['function']()

class MultiPageMenu: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self,title:str,pages:list,icons:list,
                funs:list,orientation : str = 'horizontal',
                default_index:int = 0, styles : dict = None) -> None:
        """
        Constructor class to generate a list which will store all our applications as an instance variable.
        @ param title (str) : title of the menu
        @ param pages (list): list of page name list
        @ param icons (list): icon list of the menus
        @ param funs (object.py,module) : script file that it is executed in each of the navigation pages
        @ param orientation (str) : orientation of the wpage
        @ param styles (dict) : styles of the navigation bar
        @ param default_index(int): default index from which you want to start the navigation
        """
        self.pages = pages
        self.icons = icons
        self.funs = funs
        self.title = title
        self.styles = self.initstyles(styles)
        self.orientation = orientation
        self.default_index = default_index  
    def initstyles(self,styles:dict):
        '''
        Initialize style
        @ param self (MultiPageMenu)
        @ param style(dict) : style to decorate the function
        Style of the object
        '''
        if styles is None:
            styles =  {
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
        return styles
    def retselected(self):
        '''
        Return selected webpage 
        @ param self (MultiPageMenu) : multipage menu object
        '''
        # 3. CSS style definitions
        selected = option_menu(self.title,self.pages, 
            icons=self.icons, 
            menu_icon="cast", default_index=self.default_index, orientation=self.orientation,
            styles=self.styles
        )
        return selected
'''
Running Class
'''
class ToRun:
    def __init__(self,dicnav:dict):
        self.dicnav = dicnav
    def run(self,selected):
        '''
        Running th webpage selected
        @ param selected (str) : selected navigation page
        '''
        self.dicnav[selected]()
    def run_data(self,selected,data:object):
        '''
        Running th webpage selected
        @ param selected (str) : Runn the navigator app selected
        '''
        self.dicnav[selected](data)
    def run_wt_input(self,selected:object,translator:object,input:str):
        """
        Corre la aplicación con un string input de entrada

        Args:
            selected (object): configuracion seleccionada
            input (str): traducción de destinop que hacer
        """        
        self.dicnav[selected](translator,input)





# '''
# Decorator to run main python function
# '''

# def 
# def dec_Multipage(main):
#     def execution(*args,**kwargs):
