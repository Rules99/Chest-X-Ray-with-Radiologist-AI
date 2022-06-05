## CONFIG FILES
def help_output(title,*args)->str:
    """
    Ayuda a visualizar la salidas 

    Args:
        title (_type_): _description_

    Returns:
        str: _description_
    """    
    title+="\n\n"
    for arg in args:
        title+=arg
        title+='\n\n'
    return title

## INPUT FILES
title_config = 'Model Selection'
sal_option_q = 'Which saliency map do you want to use?'
model_option_q = 'Which model do you want to use?'
color_option_q = 'Which color do you want to use?'
alpha_option_s = 'Select an alpha option'
threshold_option_s = 'Select a threshold option'
## DOC FILES
help_sal_q = 'Use different saliency maps to capture pneumonia or normal controls'


# help_model_q = '''
#                 Choose the training model you want to use to perform the diagnosis:
#                 \n
#                 \n

#                 ALL : pre-training on a mixture of databases.
#                 \n
#                 Radiological Society of North America: pre-training in the RSNA database.
#                 \n
#                 National Institutes of Health: pre-training on the NIH database.       
#                 \n
#                 PadChest: pre-training on the PadChest database.                            
#                 \n
#                 Chexpert: pre-training on the Chexpert database.                            
#                 \n
#                 Mimic: pre-training in the MIMIC IV database.
#                 '''
# HELPER FOR MODEL OPTIONS

all_st = 'ALL : pre-training on a mixture of databases.'
rsna_st =   'Radiological Society of North America: pre-training in the RSNA database.'
nih_st = 'National Institutes of Health: pre-training on the NIH database.'
pc_st = 'PadChest: pre-training on the PadChest database.'
chex_st = 'Chexpert: pre-training on the Chexpert database.'
mimic_st = 'Mimic: pre-training in the MIMIC IV database.'
title_mod_help = 'Choose the training model you want to use to perform the diagnosis:'
l_model_sentences = [all_st,rsna_st,nih_st,pc_st,chex_st,mimic_st]
help_model_q = help_output(title_mod_help,*l_model_sentences)


# HELPER FOR SALIENCY MAPS
gradcam_st = 'GradCAM: The classical saliency map'
gradcam_plus_st = 'GradCAMPlusPlus : A more sofisticated algorithm'
xgradcam_st = 'XGradCAM: Takes into account the pathologies in a more balanced way'
s_model_sentences = [gradcam_st,gradcam_plus_st,xgradcam_st]
title_sal_help = 'Choose the saliency map  you want to use to perform the diagnosis:'
help_sal_q = help_output(title_sal_help,*s_model_sentences)



# Color option
help_color_q = 'Select the heatmap type you want to use'

# Alpha option
help_alpha_s = 'Select to which intensity you want to visualize the saliency map, the greater is the value the more intense is the color'
# Threshold option
help_threshold_s = 'Select the most attended zones the model takes, the greater is the value the more important is the zone to diagnose the class'


## OUTPUT PREDICTION
# or_im_output = 'Original Image'
pr_im_output = 'Prediction Model Attention'


