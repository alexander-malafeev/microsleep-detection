# Automatic Microsleep Detection With Deep Learning

This repository contains the code we used in our paper "Automatic Microsleep Detection with Deep Learning".

Citation:

https://arxiv.org/abs/2009.03027

"Automatic detection of microsleep episodes with deep learning"
Alexander Malafeev, Anneke Hertig-Godeschalk, David R. Schreier, Jelena Skorucak, Johannes Mathis, Peter Achermann

## Getting Started

The code in this repository might be improved  so if you would like to have exactly the code we used in the  paper
please revert to the first commit.
The dataset is available at https://zenodo.org/record/3251716.
For the convenience of downloading I prepared the script ./data/download.sh. 
You can also train the model using your own data. 

### Prerequisites

You will need a GPU for training and inference.

### Installing

First you would need to install Python 3, Keras and Tensorflow to run the code.

In order to run script ./data/download.sh you would need Bash. You can also download the data
manually and put  mat files into the folder ./data/files.


## Structure
Code for models is located in 3 folders: CNN, CNN_LSTM and embeddings_128_CNN_16s.
### CNN 
CNN folder contains multiple subfolders which stand for different models we investigated:
CNN_2s, CNN_4s, CNN_8s, CNN_16s, CNN_32s contain networks which use EEG channel and two EOG channels with 
corresponding length of the sliding window.
CNN_16s_u differs from previously described networks by using uniform weights for classes and 
CNN_8s_1ch contains model which uses a single EEG channel and 8s long sliding window.


## Author

* **Alexander Malafeev** 

## License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details


