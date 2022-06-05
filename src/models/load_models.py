# from tensorboard import summary
import torchxrayvision as xrv
# from torchxrayvision.models import model_urls
import torch
import torch.nn as nn
import pickle
import pandas as pd
import os





bin_ckpoints = { 
                'densenet121-res224-all'     : 'ALL'                                    ,
                'densenet121-res224-rsna'    : 'Radiological Society of North America'  ,
                'densenet121-res224-nih'     :  'National Institutes of Health'         , 
                'densenet121-res224-pc'      :  'PadChest'                              ,
                'densenet121-res224-chex'    :  'Chexpert'                              ,
                'densenet121-res224-mimic_ch':  'Mimic IV'   
               }
binckpoints = os.listdir('models/checkpoints/binclass')


premultckpoints = ['densenet121-res224-all','densenet121-res224-mimic_ch','densenet121-res224-mimic_ch']
multickpoints = os.listdir('models/checkpoints/multiclass')


def checkcuda():
    """
    Check if cuda is available or not
    Returns:
        _type_: _description_
    """    
    return 'cuda' if torch.cuda.is_available() else 'cpu'
def devicecuda():
    """
    Check the device in which you are loading the data
    """    
    return torch.device(checkcuda())

def loadmetadata(metadatapath:str,weights:str = None):
    """
    Load metadata realted to the results of the model

    Args:
        metadatapath (str): path of the pickle metadata
        weights (str, optional): weights of the model. Defaults to None.
    """   
    # Load data (deserialize)
    with open(metadatapath, 'rb') as handle:
        metadata = pickle.load(handle)
    modeldf = pd.DataFrame({'models':metadata['models'],
    'opt_thresh':metadata['opt_thresh']})
    opt_threshold = modeldf[modeldf['models'] == weights]['opt_thresh'].values[0]
    return opt_threshold



def definemodel(weights:str = "densenet121-res224-mimic_nb",device:torch.device = None)->object:
    """
    Definition of the model
    Args:
        weights (str, optional): Weights of the pretrained model. Defaults to "densenet121-res224-mimic_nb".
        weights (str, optional): Device where pretrained model wants to be loaded. Defaults to True.
    Returns:
        object: model object
    """       
    ## Load the mdodel
    # model = xrv.models.DenseNet(weights=weights)
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
    ### Moodifications of the model
    model.op_threshs = None # prevent pre-trained model calibration
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential( nn.Linear(in_features=num_ftrs, out_features=2),torch.nn.LogSoftmax(dim=1)) #  Change the linear layer since 18 outputs to 2 
    model = model.to(device)
    
    return model

def definemltmodel(weights:str = "densenet121-res224-mimic_nb",device:torch.device = None,pretrained:bool = False)->object:
    """
    Definition of the multi model
    Args:
        weights (str, optional): Weights of the pretrained model. Defaults to "densenet121-res224-mimic_nb".
        weights (str, optional): Device where pretrained model wants to be loaded. Defaults to True.
    Returns:
        object: model object
    """       
    ## Load the mdodel
    model = xrv.models.DenseNet(weights=weights)
    ### Moodifications of the model
    model.op_threshs = None # prevent pre-trained model calibration
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_ftrs, out_features=14) #  Change the linear layer since 18 outputs to 2 
    model = model.to(device)
    return model



def load_bin_model(weights:str = 'densenet121-res224-all', device:torch.device = None):
    """
    Load the model trained for pneumonia detection
    Args:
        weights (str, optional): weights of the pretrained model. Defaults to 'densenet121-res224-all'.
        cuda (bool, optional): if use cuda. Defaults to True.
    """    
    
    model = definemodel(weights=weights,device= device)
    path_model = weights.split('-')[-1]
    base_name = f'./models/checkpoints/binclass/{path_model}.pt'
    model.load_state_dict(torch.load(base_name,map_location = device))
    return model

def load_mlt_model(weights:str = 'densenet121-res224-mimic_nb' ,
                   device:torch.device = None):
    model = definemltmodel(weights=weights,device=device)
    model.load_state_dict(torch.load(f'./models/checkpoints/multiclass/checkpoint_ch_ce.pt',map_location = device))
    return model


def loadmetadatamlt(metadatapath:str,weights:str = "densenet121-res224-mimic_nb"):
    """
    Load metadata realted to the results of the model

    Args:
        metadatapath (str): path of the pickle metadata
        weights (str, optional): weights of the model. Defaults to None.
    """   
    # Load data (deserialize)
    with open(metadatapath, 'rb') as handle:
        metadata = pickle.load(handle)
    return metadata



    
    

