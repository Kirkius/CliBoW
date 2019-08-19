# Clinical Bag-of-Words
This repository contains the code written for the final thesis during my Masters Medical Informatics. This project took place between November 2018 and July 2019 at the departement of Biomedical Engineering & Physics, Amsterdam UMC - Locatie AMC.

Much of the code still needs to be cleaned up, and many functions still require documentation. I will try to finish these up in my spare time.

The data used for this project came from the [MIMIC-III database](https://mimic.physionet.org). This data is not present in this repository.

All code was written in Python 3.6.

## Content
The main file calls all the necessary functions required for a single run. The BagofWords file contains all the functions needed for the Bag-of-Words feature extraction and the functions for training and testing the model. The Visualize_functions file contains all the functions to needed to visualize the results.

Additionally, the folder Preprocess contains the functions used to clean the data downloaded from MIMIC-III, and the folder SaveLoad_Functions contains the functions used for the saving and loading of data and models.
