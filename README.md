## Project Overview

This is the fourth project of the Udacity Deep Learning Nanodegree. In this project, I built a TV scripts generatorusing RNNs. Trained on part of the Seinfeld 
dataset of scripts from 9 seasons, the RNN can generate a new, "fake" TV script.

## Training Dataset

Model training in this project uses the [Seinfeld Chronicles dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv), which is publicly 
available via Kaggle. This dataset is ~3.4MB.

## Model Structure

The model uses recurrent neural networks (RNN) and consists of an embedding layer (with 256 embedding dimensions), a 2-layer LSTM (with 512 hidden dimensions),
and, finally, a fully-connected layer. With 15 epochs of training, the RNN was able to achieve ~2.7-2.8 in training loss. Lastly, the RNN can generate "fake"
TV scripts of any specified length when a prime word is given.

## Python Package Dependencies

* [PyTorch](https://pytorch.org/)
* [NumPy](https://numpy.org/install/)

## Run Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/chloeh13q/DLND-TV-Script-Generation
		cd DLND-TV-Script-Generation
	```
2. Make sure you have already installed the necessary Python packages.
3. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dlnd_tv_script_generation.ipynb
	```
