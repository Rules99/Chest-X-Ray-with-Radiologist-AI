# Chest-X Ray with Radiologist AI
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Python version](https://img.shields.io/badge/python-3.8.13-blue.svg)](https://pypi.org/project/kedro/)
[![License](https://img.shields.io/github/license/TezRomacH/python-package-template)](https://github.com/TezRomacH/python-package-template/blob/master/LICENSE)

`cxrai` is a deep learning system in production tool to make Chest-X-Ray diagnosis

<img src="./docs/doctors.jpg"
     alt="CX-AI Icon"
     style="text-align: center; margin-right: 10px;" />
-----------------

# How to use it?

## Project description
Project description

## Code execution

```sh
streamlit run app.py
```

## Streamlit application

--------
# Development

## Installation instructions

Run the silent installation of Miniconda/Anaconda in case you don't have this software in your environment.

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

Once you have installed Miniconda/Anaconda, create a Python 3.7 environment.

```sh
conda create --name cxr-rai python=3.8.13
conda activate cxr-rai
```

Clone this repository and install it inside your recently created Conda environment.

```sh
git clone https://github.com/Rules99/Chest-X-Ray-with-Radiologist-AI
cd Chest-X-Ray-with-Radiologist-AI
pip install -r requirements.txt
```

## Dependencies installation
- python 3.8.13
- efficientnet 1.1.1
- gensim 3.8.3
- googletrans 4.0.0-rc1
- grad-cam 1.3.7
- h5py 3.1.0
- imgaug 0.4.0
- matplotlib 3.5.1
- nltk 3.4.5
- numpy 1.19.5
- opencv-python-headless 
- pandas   1.4.2
- plotly   5.8.0
- requests   2.27.1
- scikit_image   0.19.2
- scikit_learn   1.0.2
- seaborn   0.11.2
- streamlit   1.8.1
- streamlit-option-menu   0.3.2
- tensorflow 2.5.3
- termcolor   1.1.0
- torch   1.11.0
- torchsummary   1.5.1
- torchvision   0.12.0
- torchxrayvision   0.0.32
- transformers   2.5.1
- tqdm 4.64.0
- Pillow   9.1.0
- protobuf   3.19.0

--------
# Authors & Contributors

cxrai was developed by:
- [Pablo Reyes](https://github.com/Rules99)
- [Fernando Pozo](www.fpozoc.com)

--------
# Acknowledgements


--------
# FAQ


--------
# Citing cxrai

```text
@misc{10.1093/nargab/lqab044,
    author = {Reyes, Pablo and Pozo, Fernando},
    title = "{Sistema de identificación e interpretación de patologías pulmonares a partir de imágenes rayos X mediante Aprendizaje Profundo}",
    year = {2022},
    month = {06},
    abstract = "{}",
    url = {}
}
```


--------
# References

