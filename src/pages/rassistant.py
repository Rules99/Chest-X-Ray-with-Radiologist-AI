from src.pages.bin_class import *
from src.pages.mlt_class import *
from src.pages.rep_gen import *



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
                

    if uploaded_file is not None:
        # Pneumonia

        with st.expander(translate_(translator,'Pneumonia Identification',dest = input),expanded=True):

            bin_neumonia(uploaded_file,
                    device = device,
                    model_option = model_option,sal_option = sal_option,color_option = color_option, 
                    alpha_option = alpha_option,threshold_option = threshold_option,
                    translator = translator,input = input)
    
        # Pathologies
        with st.expander(translate_(translator,'Pathologies Identification',dest = input),expanded=True):

            mlt_neumonia(
                        uploaded_file,
                        device = device,
                        sal_option = sal_option,color_option = color_option, 
                        alpha_option = alpha_option,threshold_option = threshold_option,
                        translator = translator,input = input)
            
        # Medical Report
            
        with st.expander(translate_(translator,'Automatic Medical Report',dest=input)):
            gen_report(uploaded_file,translator,input)
