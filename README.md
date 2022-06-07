---
title: Chest-X Ray with Radiologist AI
emoji: 👨‍⚕️🦴
colorFrom: black
colorTo: white
sdk: streamlit
app_file: app.py
pinned: false
---
# Configuration
`title`: _string_  
Display title for the Space
`emoji`: _string_  
Space emoji (emoji-only character allowed)
`colorFrom`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)
`colorTo`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)
`sdk`: _string_  
Can be either `gradio` or `streamlit`
`sdk_version` : _string_  
Only applicable for `streamlit` SDK.  
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

`pinned`: _boolean_  
Whether the Space stays on top of your list.





<img src="./docs/doctors.jpg"
     alt="CX-AI Icon"
     style="text-align: center; margin-right: 10px;" />
-----------------

test

# Chest-X Ray with Radiologist AI: production tool to make Chest-X-Ray Diagnosis
-----------------

The use of software tools in medicine based on machine learning (ML) techniques has already shown a direct impact on the diagnosis in different clinical application areas. These techniques will likely help to provide better healthcare. In the same way, the revolution of personalized medicine requires the emerging analysis of many imaging-related data. This project's primary purpose would be to detect anomalies, conditions or pathologies in image tests like chest-X rays with ML-based techniques. We will explore the use and combination of other data types like the available collection of Electronic Health Records (EHR) of every single patient to develop novel treatments and identify patients who will benefit from them. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
Instrucciones de instalación
------------
Notas: nameenv puede ser el nombre de entorno que tu prefieras, lo único que siempre debe ir junto.

ESCRIBIMOS EN ANACONDA PROMPT LOS SIGUIENTES COMANDOS

1. Crea un nuevo entorno de conda en python 3.8.13:

```sh
conda create -n nameenv python=3.8.13
```

2. Activa el entorno establecido:


```sh
conda activate nameenv
```

3. Instala los paquetes a requerir en el entorno:

```sh
pip install -r requirements.txt
```
Ejecución del código
------------

```sh
streamlit run app.py
```

Dependencias 
------------
- Python 3.8.13
