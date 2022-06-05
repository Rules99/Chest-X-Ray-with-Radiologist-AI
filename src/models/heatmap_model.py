
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM



### Saliency Maps globals
listsalmaps = ['GradCAM', 'GradCAMPlusPlus', 'XGradCAM']
diccolors = {
        cv2.COLORMAP_AUTUMN : 'autumn',
        cv2.COLORMAP_BONE : 'bone',
        cv2.COLORMAP_JET : 'jet',
        cv2.COLORMAP_WINTER : 'winter',
        cv2.COLORMAP_RAINBOW : 'rainbow',
        cv2.COLORMAP_OCEAN : 'ocean',
        cv2.COLORMAP_SUMMER : 'summer',
        cv2.COLORMAP_SPRING : 'spring',
        cv2.COLORMAP_COOL : 'cool',
        cv2.COLORMAP_HSV : 'hsv',
        cv2.COLORMAP_PINK : 'pink',
        cv2.COLORMAP_HOT : 'hot',
        cv2.COLORMAP_MAGMA : 'magma',
        cv2.COLORMAP_INFERNO : 'inferno',
        cv2.COLORMAP_PLASMA : 'plasma',
        cv2.COLORMAP_VIRIDIS : 'viridis',
        cv2.COLORMAP_CIVIDIS : 'cividis',
        cv2.COLORMAP_TWILIGHT : 'twilight',
        cv2.COLORMAP_TURBO : 'turbo',
}
dicnames = {value: key for (key,value) in diccolors.items()}


def show_camper_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colorname: str = 'jet',
                      alpha:float = 1,
                      threshold:float = 1) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    colormap = dicnames[colorname]
    # Prune the mask
    threshold = threshold*np.max(mask)
    mask = np.piecewise(mask,[mask<threshold,mask>=threshold],[lambda x: 0,lambda x: x])
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    
    
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = alpha*heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def makeheatmap(input_tensor:torch.Tensor,
                or_im:np.array = None,
                model:object = None,
                gradcam_model : str = 'GradCAM',
                colorname :str = 'jet',
                alpha:float = 1,
                threshold:float = 1,
                maximize:int = 1,
                typedevice : str = None)->np.array:

    target_layers = [model.features.denseblock4[i] for i in range(len(model.features.denseblock4))]
    # # Construct the CAM object once, and then re-use it on many images:
    cam = eval(gradcam_model)(model=model, target_layers=target_layers, use_cuda=False)

    # # You can also use it within a with statement, to make sure it is freed,
    # # In case you need to re-create it inside an outer loop:
    # # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    # #   ...

    # # We have to specify the target we want to generate
    # # the Class Activation Maps for.
    # # If targets is None, the highest scoring category
    # # will be used for every image in the batch.
    # # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(maximize)]

    # # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    # Depending on the size you must use one or another numpy mapping
    if typedevice == 'cuda':
        img_vis = input_tensor[0,:,:][...,None].detach().cpu().numpy().repeat(3,-1)
    elif typedevice == 'cpu':
        img_vis = input_tensor[0,:,:][...,None].detach().numpy().repeat(3,-1)
    visualization = show_camper_on_image(img_vis, grayscale_cam, use_rgb=True,colorname=colorname,alpha=alpha,threshold=threshold)
    # # In this example grayscale_cam has only one image in the batch:
    return visualization


def plot_color_gradients(category:str, cmap_list:list,
                         diagnosis:str = 'normal',translator:object = None, input: str = None,labels:list = None,
                         width:int = 15, height:int = 12,fontsize:int = 10):
    """
    Visualiza la importancia de las zonas de los mapas de saliencia

    Args:
        category (str): Nombre del título de escala
        cmap_list (list): lista dce escalas de color 
        diagnosis (str): predicción del diagnóstico
        translator (object): traductor del predictor
        input (str) : destino a clasificar
        width (int, optional): anchura de la paleta. Defaults to 15.
        height (int, optional): altura de la paleta. Defaults to 12.
        fontsize (int, optional): tamaño de la fuente. Defaults to 10.
    """    
    # Create figure and adjust figure height to number of colormaps
    cmaps = {}
    # Crear un linspace del o al 255
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))


   
    # CRear una figura de cierto tamaño
    figh = 0.35 + 0.15 + (height + (height - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=height + 1, figsize=(width, height))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category}', fontsize=14)
    # Set of description
    [weak,mid,imp,est] = labels
   

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        # ax.text(-0.01, 0.5, name, va='top', ha='right', fontsize=fontsize,
        #         transform=ax.transAxes)
                # Labels
        ax.text(0.25, -0.1, weak, va='top', ha='right', fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(0.5, -0.1, mid, va='top', ha='right', fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(0.75, -0.1, imp, va='top', ha='right', fontsize=fontsize,
                transform=ax.transAxes)
        ax.text(1, -0.1, est, va='top', ha='right', fontsize=fontsize,
                transform=ax.transAxes)
    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list
    return fig