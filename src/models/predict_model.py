  
from matplotlib import image
import torch
import torch.nn.functional as F
import numpy as np

def forecast_bin_model(image:torch.Tensor,model:object = None,device:torch.device = None,opt_thresh:float = None):
    image = image.to(device)
    # Forward pass of the model 
    with torch.torch.no_grad():
        outputs = model(image)
        probs = torch.exp(outputs)
        # Decide according to some threshold
        predictions = 1 if probs.detach().numpy()[0,1] > opt_thresh else 0
        
        if predictions == 1 :
            prob = probs.detach().numpy()[0,1]
        else:
            prob = probs.detach().numpy()[0,0]
        # _, predictions = torch.max(outputs.data, 1)
    return prob,predictions


def forecast_mlt_model(image:torch.Tensor,model:object = None,device:torch.device = None,opt_thresh:float = None):
    # Forward pass of the model 
    with torch.torch.no_grad():
        outputs = model(image)
        
        probs = F.sigmoid(outputs).numpy().flatten()
        opt_thresh = opt_thresh.to_numpy()
        condit = (probs >= opt_thresh)

        
        # Decide according to some threshold
        predictions = 1*condit
    
        
        # _, predictions = torch.max(outputs.data, 1)
    return probs,predictions


    