# from fnmatch import translate
# from shutil import unregister_unpack_format

from PIL import ImageOps, Image
import plotly.express as px
import streamlit as st
from tqdm import tnrange
from src.models.load_models import load_mlt_model
from src.data.make_dataset import preprocess
from src.models.load_models import *
from src.models.heatmap_model import *
from src.models.predict_model import *
from src.utils.fun import *
from src.utils.traductor.translation import translate_
from src.utils.traductor.strings_pathologies import *
from src.utils.traductor.general_strings import *
from src.utils.traductor.strings_neumonia import sal_option_q,help_sal_q,color_option_q,help_color_q,alpha_option_s,help_alpha_s,threshold_option_s,help_threshold_s



def mlt_neumonia(uploaded_file:object,
                 device:object = None,
                 sal_option:str = 'GradCAM',color_option:str = 'jet', 
                 alpha_option:float = 1,threshold_option:float = 0,
                 translator:object = None,input = 'en')->None:
    """
    Función para clasificar de forma binaria la neumonía

    Args:
        uploaded_file (object): archivo a subir
        device (object, optional): correr el modelo en cpu o en cuda. Defaults to None.
        sal_option (str, optional): opcion del mapa de saliencia. Defaults to 'GradCAM'.
        color_option (str, optional): opcion del color. Defaults to 'jet'.
        alpha_option (float, optional): opcion para cambiar la intensidad. Defaults to 1.
        threshold_option (float, optional): opcion para reconocer las zonas mas importantes. Defaults to 0.
        translator (object, optional): objeto para traducir los strings. Defaults to None.
        input (str, optional): destino de traduccion. Defaults to 'en'.
    """        
    
 
    or_im = ImageOps.grayscale(Image.open(uploaded_file))
    image = preprocess(or_im)
    
    


    model = load_mlt_model(device = device )
    # Predict the output
    metadata = loadmetadatamlt('models/pickles/opt_mlt_thresh.pickle')
    names = metadata['name']
    opt_thresh = metadata['threshold']
    probabilities,predict_classes = forecast_mlt_model(image,model=model,device=device,opt_thresh=opt_thresh)
    
    
    
   


    # Patologias detectadas
    diagnostics = []
    visuals = []
    pathology_ = translate_(translator,'Pathology',dest=input)
    probability_ = translate_(translator,'Probability',dest=input)
    i = 0

    # Get the heatmap of rach pathology and the metadata results
    for (name,predict_class,prob) in zip(names,predict_classes,probabilities):
        # Visualizar cada una de las patologías en el caso de que se manifiesten
        if predict_class == 1:
            
                visualization = makeheatmap(image,
                        model = model,
                        gradcam_model  = sal_option,
                        colorname = color_option,
                        alpha = alpha_option,
                        threshold=threshold_option,
                        maximize = i,
                        typedevice = checkcuda()
                )
                # Traducir las palabras
                identi_pathology = translate_(translator,f"{name}",dest=input).capitalize()
                prob_round = round(float(prob),2)
                text_pathology = f"{pathology_}: {identi_pathology}.  {probability_}: {prob_round}"
                            
                # Append metadata and heatmapo to lists
                diagnostics.append(text_pathology)
                visuals.append(visualization[0,:,:])
                i+=1
    # Create an array of (P,3,N,M) -- > P is the number of possible pathologies
    visuals = np.array(visuals)
   
   
            
    
    # Make the heatmap depending on the output target
    
    col1,col2 = st.columns((1,1))
    with col1:
        fig = px.imshow(or_im,binary_string = True)
        fig.update_layout(title_text=translate_(translator,or_im_output,dest=input),title_x=0,margin= dict(l=0,r=0,b=10,t=30), yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig,use_column_width=True)
    with col2:
        # Pathology visualizations
        fig2 = px.imshow(visuals,animation_frame= 0,labels=dict(animation_frame='pathology'))
        fig2.update_layout(title_text=diagnostics[0],title_x=0,margin= dict(l=0,r=0,b=10,t=30), yaxis_title=None, xaxis_title=None)
        for i, frame in enumerate(fig2.frames):
            frame.layout.title = diagnostics[i]
            frame.layout.margin = margin= dict(l=0,r=0,b=10,t=30)
        st.plotly_chart(fig2)
        # Plot palette
        labels = [translate_(translator,label,dest = input).capitalize() for label in pats('pathology')]
        fig3 = plot_color_gradients(
                            translate_(translator,possible_pat('pathology'),dest = input), 
                            cmap_list = [color_option],
                            diagnosis = 'pathology',
                            labels= labels ,
                            width = 12, height = 10,fontsize = 10)
        st.pyplot(fig3,use_column_width = True)



   

def app(translator,input):
    # Site bar to run the model
    with st.sidebar:
            st.write('\n')
            markdown_(translate_(translator,title_config,dest=input) ,'h2','center')
            
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
    uploaded_file =   st.file_uploader(f'{translate_(translator,input_option_q,dest=input)}')
    # Preprocess file

    if uploaded_file is not None:
        mlt_neumonia(uploaded_file,
                    device = device,
                    sal_option = sal_option,color_option = color_option, 
                    alpha_option = alpha_option,threshold_option = threshold_option,
                    translator = translator,input = input)
        
    