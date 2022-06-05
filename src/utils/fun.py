import streamlit as st
## FUNCTIONS
def markdown_(text:str,tag:str = 'h3',style:str ='center',unsafe_allow_html:bool = True):
    '''
    Customize the markdown in a fast way
    @ param text (str): text of the file
    @ param tag (str): html attr of the text
    @ param style (str): location of the text file
    @ param unsafe_allow_html (bool): allow html in an unsafe way
    '''
    return st.markdown(f'''
                       <{tag} style ='text-align: {style};'>{text}</{tag}>
                       ''',unsafe_allow_html = unsafe_allow_html )
def markdown_link(text:str,tag:str,text_align:str,link:str,unsafe_allow_html:bool = True ):
    '''
    Display link of the markdown list
    @ param text (str): text of the file
    @ param tag (str): html attr of the text
    @ param text_align (str): location of the text file
    @ param link (str) : link if the markdown
    @ param unsafe_allow_html (bool): allow html in an unsafe way
    '''
    return st.markdown(
            f"""<{tag}><a style='display: block; text-align: {text_align};' href="{link}">{text}</a></{tag}>
            """,
            unsafe_allow_html=unsafe_allow_html
            )

def instanceterms(*args,**kwargs):
    return args,kwargs