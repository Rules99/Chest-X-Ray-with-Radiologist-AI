
import streamlit as st
from torch import binary_cross_entropy_with_logits
import plotly.express as px

from PIL import ImageOps, Image
### Source modules
from src.data.make_dataset import preprocess
from src.models.load_models import *
from src.models.heatmap_model import *
from src.models.predict_model import *
from src.utils.fun import *
from src.utils.traductor.translation import translate_
from src.utils.traductor.strings_neumonia import *
from src.utils.traductor.general_strings import *


def bin_neumonia(uploaded_file:object,
                 device:object = None,
                 model_option:str = None,sal_option:str = 'GradCAM',color_option:str = 'jet', 
                 alpha_option:float = 1,threshold_option:float = 0,
                 translator:object = None,input = 'en')->None:
    """
    Función para clasificar de forma binaria la neumonía

    Args:
        uploaded_file (object): archivo a subir
        device (object, optional): correr el modelo en cpu o en cuda. Defaults to None.
        model_option (str, optional): opcion del modelo binario. Defaults to None.
        sal_option (str, optional): opcion del mapa de saliencia. Defaults to 'GradCAM'.
        color_option (str, optional): opcion del color. Defaults to 'jet'.
        alpha_option (float, optional): opcion para cambiar la intensidad. Defaults to 1.
        threshold_option (float, optional): opcion para reconocer las zonas mas importantes. Defaults to 0.
        translator (object, optional): objeto para traducir los strings. Defaults to None.
        input (str, optional): destino de traduccion. Defaults to 'en'.
    """        
    
    
    ### To gray scale
    or_im = ImageOps.grayscale(Image.open(uploaded_file))
    image = preprocess(or_im)
        # Select the model you want
    
    
    model = load_bin_model(model_option,device =device)
    # Predict the output
    opt_thresh = loadmetadata('models/pickles/opt_bin_thresh.pickle',model_option)
    prob,predict_class = forecast_bin_model(image,model=model,device=device,opt_thresh=opt_thresh)
    
    # Make the heatmap depending on the output target
    
    visualization = makeheatmap(image,
            or_im=None,
            model = model,
            gradcam_model  = sal_option,
            colorname = color_option,
            alpha = alpha_option,
            threshold=threshold_option,
            maximize = predict_class,
            typedevice = checkcuda()
    )
    # Predicción
    name_prediction = translate_(translator,'Pneumonia' if predict_class == 1 else 'Normal',dest=input)
    __, __, m3, __, __ = st.columns((1,1,1,1,1))
    m3.metric(f"{translate_(translator,'Diagnosis',dest=input)}", name_prediction, round(float(prob),2))
    

    col1,col2 = st.columns((1,1))
    with col1:
        fig = px.imshow(or_im,binary_string = True)
        fig.update_layout(title_text=translate_(translator,or_im_output,dest=input),title_x=0,margin= dict(l=0,r=0,b=10,t=30), yaxis_title=None, xaxis_title=None)
        # markdown_(translate_(translator,or_im_output,dest=input),'h3','center')

        st.plotly_chart(fig,use_column_width=True)
    with col2:
        fig2 = px.imshow(visualization[0,:,:,:],color_continuous_scale=color_option,binary_string=False)
        fig2.update_layout(title_text=translate_(translator,pr_im_output,dest=input),title_x=0,margin= dict(l=0,r=0,b=10,t=30), yaxis_title=None, xaxis_title=None)
        
        # Make prediction visualization
        
        st.plotly_chart(fig2,use_column_width = True)

        # Plot palette
        labels = [translate_(translator,label,dest = input) for label in pats(name_prediction)]
    
        fig3 = plot_color_gradients(
                            translate_(translator,possible_pat(name_prediction),dest = input), 
                            cmap_list = [color_option],
                            diagnosis = name_prediction,
                            labels= labels ,
                            width = 12, height = 10,fontsize = 10)
        st.pyplot(fig3,use_column_width = True)
        



def app(translator:object,input:str):
    # Site bar to run the model
    with st.sidebar:
            st.write('\n')
            markdown_(f'{translate_(translator,title_config,dest=input)}','h2','center')
            # Translate help model q
            # help_model_q = help_output(translate_(translator,title_mod_help,dest=input),
                                    #   *[translate_(translator,l_model,dest=input) for l_model in l_model_sentences])
            model_option = st.selectbox(
                f'{translate_(translator,model_option_q,dest=input)}',
                bin_ckpoints.keys(),index =0,format_func= lambda x: bin_ckpoints[x],help=help_model_q)
            
            sal_option = st.selectbox(
                f'{translate_(translator,sal_option_q,dest=input)}',
                listsalmaps,index =0,help=help_sal_q)
            
            color_option = st.selectbox(
                f'{translate_(translator,color_option_q,dest=input)}',
                dicnames.keys(),index =list(diccolors.values()).index('jet'),help =help_color_q)

            alpha_option = st.slider(
                    f'{translate_(translator,alpha_option_s,dest=input)}',
                    0.0, 1.0, step = 0.01,value = 1.0,help =help_alpha_s)
        
            threshold_option = st.slider(
                    f'{translate_(translator,threshold_option_s,dest=input)}',
                    0.0, 1.0, step = 0.01,value = 0.0,help =help_threshold_s)
            
            
            
       
    device = devicecuda()
    # INput a file
    uploaded_file = st.file_uploader(f'{translate_(translator,input_option_q,dest=input)}')
    # Preprocess file

    if uploaded_file is not None:
        # Modelo pneumonia
        bin_neumonia(uploaded_file,
                    device = device,
                    model_option = model_option,sal_option = sal_option,color_option = color_option, 
                    alpha_option = alpha_option,threshold_option = threshold_option,
                    translator = translator,input = input)
    