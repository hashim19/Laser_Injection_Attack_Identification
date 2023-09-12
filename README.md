# Laser_Injection_Attack_Identification
This repository contains code for laser injection attack detection.

## Installation
Clone this repository to your workspace using the following command.

`git clone https://github.com/hashim19/Laser_Injection_Attack_Identification.git`

Run the following command to go inside the parent directory of the repository.

`cd Laser_Injection_Attack_Identification`

### Follow this [tutorial](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) to create a virtual environment in the parent directory of this repository

Activate the virtual environment using the following commands. 

`source *environment name* /bin/activate`

Once, the environment is activated, run the following command in the parent directory of this repository to install the required libraries.

`pip3 install -r requiremnts.txt`

## Dataset
The dataset used to test the algorithms mentioned below is available [here](https://www.kaggle.com/datasets/hashimali19/laser-injection-data).

Download the dataset to a local directory.

## Usage

### extract_features.py: 
Extract 4 types of features and save them in npy format in the audio_features directory. This file can also train an SVM model or a GMM model. Toggle train_SVM or train_GMM to True to train the SVM or GMM model respectively. The following types of features can be extracted.

1. [CQCC] (https://ieeexplore.ieee.org/document/8659537)
2. [DWT] (https://pywavelets.readthedocs.io/en/latest/)
3. [LFCC] (https://spafe.readthedocs.io/en/latest/features/lfcc.html)
4. [MFCC] (https://librosa.org/doc/0.10.1/index.html)

The following variables need to be set before the features can be extracted, 

1. **audio_dir:** *Provide the path to the parent directory of the dataset Here*.
2. **train_GMM:** *False* if only extracting features
3. **train_SVM:** *False* if only extracting features.

After configuring these variables, you can run the following command to extract features,

`python3 extract_features.py`

if **train_GMM** or **train_SVM** is true, a GMM or SVM model will be trained respectively and saved to the models directory.

### generate_scores.py: 

This file generates the scores of the test audio using the trained models and saves the scores in the audio_features directory. Path to the dataset parent directory and audio_features folder needs to be provided. 

Run `python3 generate_scores.py` to generate scores of trained models on the test audio.
