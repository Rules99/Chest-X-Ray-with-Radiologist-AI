# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import shutil
import random
import os
import itertools
import collections
from termcolor import colored
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from IPython.display import clear_output

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def grafica_entrenamiento(tr_acc=[], val_acc=[], tr_loss=[], val_loss=[], best_epoch=0,
                          figsize=(10,8)):
    plt.figure(figsize=figsize)
    if len(val_acc)>0:
        ax = plt.subplot(1,2,1)
        plt.plot(1+np.arange(len(tr_acc)),  100*np.array(tr_acc))
        plt.plot(1+np.arange(len(val_acc)), 100*np.array(val_acc))
        plt.plot(1+best_epoch, 100*val_acc[best_epoch], 'or')
        plt.title('tasa de acierto del modelo (%)', fontsize=18)
        plt.ylabel('tasa de acierto (%)', fontsize=18)
        plt.xlabel('época', fontsize=18)
        plt.legend(['entrenamiento', 'validación'], loc='upper left')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax = plt.subplot(1,2,2)
    else:
        ax = plt.subplot(1,1,1)

    plt.plot(1+np.arange(len(tr_loss)), np.array(tr_loss))
    plt.plot(1+np.arange(len(val_loss)), np.array(val_loss))
    plt.plot(1+best_epoch, val_loss[best_epoch], 'or')
    plt.title('loss del modelo', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.xlabel('época', fontsize=18)
    plt.legend(['entrenamiento', 'validación'], loc='upper left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0,1])
    plt.show()

    
def grafica_history(history):
          '''
          Graph the history 
          @param history (keras.Model) : Model containing the information of the training data

          '''
          # list all data in history
          print(history.history.keys())
          # summarize history for accuracy
          plt.plot(history.history['accuracy'])
          plt.plot(history.history['val_accuracy'])
          plt.title('model accuracy')
          plt.ylabel('accuracy')
          plt.xlabel('epoch')
          plt.legend(['train', 'test'], loc='upper left')
          plt.show()
          # summarize history for loss
          plt.plot(history.history['loss'])
          plt.plot(history.history['val_loss'])
          plt.title('model loss')
          plt.ylabel('loss')
          plt.xlabel('epoch')
          plt.legend(['train', 'test'], loc='upper left')
          plt.show()
""
def is_image(filename, verbose=False):
    """
    Check si hay archivos que no son imagenes
    """
    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    return False

""
def extraer_img_labels(val_ds):
    """
    Función para extraer las imágenes y el target de un dataset de TF.
    """
    images_val= []
    y_val = []
    for images, labels in val_ds.take(-1):  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        images_val.append(numpy_images)
        y_val.append(numpy_labels)
    images_val = np.array(list(itertools.chain(*images_val)))
    y_val = np.array(list(itertools.chain(*y_val)))
    return images_val, y_val

""
def conf_matrix(y_val, y_pred, threshold = 0.5):
    print("Confusion Matrix con threshold:",threshold)
    y_pred2 = (y_pred > threshold)
    matrix = confusion_matrix(y_val, y_pred2)
    df_cm = pd.DataFrame(matrix, range(2), range(2))
    sns.set(font_scale=1.4)
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in
              zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(df_cm, annot=labels, fmt='', annot_kws={"size": 16})

    plt.show()

""
def curva_roc(y_val, y_pred,epochs, model_name):
    
    sns.set_style("white")
    fpr, tpr, thresholds_keras = roc_curve(y_val, y_pred)
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(fpr, tpr, 'k-', label = str(epochs) + ' epochs (%2.2f)' % auc(fpr, tpr))
    ax1.plot(fpr, fpr, 'b--', label = 'Random Guess')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    plt.title("ROC COVID "+model_name)
    ax1.legend();
    plt.show()
""
def make_confusion_matrix(y_val, y_pred, threshold,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=False,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=  (9,6),
                          cmap='YlOrBr',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    y_pred2 = (y_pred > threshold)
    matrix = confusion_matrix(y_val, y_pred2)
    cf = pd.DataFrame(matrix, range(2), range(2))
    cf = cf.to_numpy()

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            threshold = threshold
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            VPN       = cf[0,0] / sum(cf[:,0])
            specifity = cf[0,0] / sum(cf[0,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision(VPP)={:0.3f}\nRecall={:0.3f}\nVPN={:0.3f}\nSpecifity{:0.3}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,VPN,specifity,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    print('Confusion Matrix with threshold = ', str(threshold))
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels, annot_kws={'size': 17},fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label' + stats_text, fontsize=18)
    else:
        plt.xlabel(stats_text, fontsize=18)
    
    if title:
        plt.title(title)
    

""
random_state = 5
test_size = 0.2
cond_neumonia = False
def getid(x):
    '''
    Returns get the ids of the patient
    x: name of the entire path file
    '''
    # return x.split("/")[-1].split("-")[0] google platform
    return x.split("/")[-1].split('\\')[-1].split("-")[0] 
def getids(povisa_path):
    '''
    Get the ids of the povisa_path 
    povisa_path : path where normal or covid patients may be saved
    '''
    return list(map(lambda x:getid(x),povisa_path))
def ret_ids(povisa_normal_path,povisa_covid_path):
    
    '''
    Return Both ids of covid and normal paths
    povisa_normal_path : path where normal patients are saved
    povisa_covid_path : path where covid patients are saved
    '''
    
    return [getids(i) for i in [povisa_normal_path,povisa_covid_path]]
def checkWellseparation(path_train,path_val,path_test):
    '''
    Check if the separation is well done
    path_train : path of the training set
    path_test  : path of the test set
    '''
    
    assert len(list(set(path_train) & set(path_val) & set(path_test)))==0 , "Can not have same images in train and validation"
def PrintInfo(train,val,test,p_train,p_val,p_test):
    '''
    
    Print some detailed information regarding the loading
    
    '''
    print(colored(f"Hay {len(train)+len(val)+len(test)}","red"))
    print(colored(f"Hay {len(p_train)+len(p_val)+len(p_test)} Povisa","blue"))
    
    print(colored(f"Hay {len(p_train)} imágenes en train y  {len(p_val)} en val y {len(p_test) } en test.","blue"))
    one_op = len(train)+len(val)+len(test)
    print(colored('________________________________________','green'))
    #######################################
    print('Imágenes repetidas en train y val: ', len(list(set(train) & set(val))))
    print(colored('________________________________________','green'))
    #######################################
    print(f'Cantidad de archivos antes de la depuración: {len(train) + len(val)+len(train)}')
################################################################################################################################### funciones_aux_extra
def checkIsimage(train,val,test):
    
    '''
    Check if the train validation image is an image
    train: training set
    val  : validation set
    '''
    # go through all files in desired folder
    for filename_train,filename_val,filename_test in zip(train,val,test):
        # check if file is actually an image file
        if not is_image(filename_train, verbose=False) :
        # if the file is not valid, remove it
            os.remove(filename_train)
        elif not is_image(filename_val, verbose=False) :
        # if the file is not valid, remove it
            os.remove(filename_val)
        elif not is_image(filename_test, verbose=False) :
        # if the file is not valid, remove it
            os.remove(filename_test)
def conteo(p_):
    '''
    Count proportion of train validation
    p_ : pathg list differentiating covid and normal
    '''
    return collections.Counter(list(map(lambda x:x.split('/')[2],p_)))
def conteotrainval(train,val,test):
    '''
    Conteo made for train and validation set
    train : path of the train dataset
    val   : validation set

    '''
    return [conteo(train),conteo(val),conteo(test)] 

def to_categorical(v):
  return np.array(list(map(lambda x:[1,0] if x==0 else [0,1],v))).astype('int')
def createDf(train):
    '''
    Create a dataframe
    train : dataframe selected (may be train or validation)
    '''
    train_df = pd.DataFrame(train, columns = ['path'])
    train_df['image'] = train_df['path'].map(lambda x: x[x.find('PATIENT'):])
    train_df['label'] = train_df['path'].map(lambda x: 'covid' if (x.find('covid')!=-1) else 'normal')
    train_df['label_num'] = train_df['label'].map(lambda x: 1 if x == 'covid' else 0).astype(int)
    x_train = train
    y_train = to_categorical(list(train_df['label_num']))
    return train_df,x_train,y_train
def trainvaltestDf(train,val,test):
    '''
    Return train and valiudation information
    '''
    return [element for sublist in [createDf(train),createDf(val),createDf(test)] for element in sublist]


def plotVisualization(conteo_train,conteo_val,conteo_test):
    '''
    Plot proportion of normal and covid patients in train and validation 
    conteo_train : proportion of normal and covid patients in train
    conteo_validation : proportion of normal and covid patients in validation
    
    '''
    fig, axs = plt.subplots(1, 3, figsize=(12,6))

    values=[]
    labels = []

    for k,v in conteo_train.items():
        values.append(v)
        labels.append(k)
    axs[0].pie(values, labels = labels, autopct=make_autopct(values), shadow=True)
    axs[0].set_title('Conjunto de train')

    values=[]
    labels = []

    for k,v in conteo_val.items():
        values.append(v)
        labels.append(k)
    axs[1].pie(values, labels = labels, autopct=make_autopct(values), shadow=True)
    axs[1].set_title('Conjunto de validación')
    
    values=[]
    labels = []

    for k,v in conteo_test.items():
        values.append(v)
        labels.append(k)
    axs[2].pie(values, labels = labels, autopct=make_autopct(values), shadow=True)
    axs[2].set_title('Conjunto de test')
    plt.show()
