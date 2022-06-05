# -*- coding: utf-8 -*-
import torchvision
from torchvision import transforms
import torch
from src.models.load_models import checkcuda

def preprocess(image:object,
                transform = torchvision.transforms.Compose([
                                        transforms.Resize(224), # maybe better to factor out resize step
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor()])):
    """
    Transform a jpg image onto 224 resolution image
    Args:
        img_path (str): path of the image
        transform (_type_, optional): Tranformation to make to the image. Defaults to torchvision.transforms.Compose([ transforms.Resize(224), # maybe better to factor out resize step transforms.CenterCrop(224), transforms.ToTensor()]).

    Returns:
        _type_: _description_
    """        

    
    if transform:
        image = transform(image)
    if checkcuda() == 'cuda':
        print('Passed')
        image = image[None,...].type(torch.cuda.FloatTensor)
    else:
        image = image[None,...].type(torch.FloatTensor)

    return image


