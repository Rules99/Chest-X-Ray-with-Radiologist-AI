
def translate_(translator:object,text:str,dest:str,src:str='en'):
    """
    Traducir objetos

    Args:
        translator (object): permite traducir objetos
        text (str): _description_
        dest (str): _description_
        src (str, optional): _description_. Defaults to 'en'.
    """    
    return translator.translate(text,src = src,dest = dest).text


