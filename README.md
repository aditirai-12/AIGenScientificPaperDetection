# Directory Strucutre
cse-472-project-ii-ai-fenerated-text-detection: This file contains the dataset being used for training, validation, and testing
main.py: contains the model implementation, training, validation, and predicting
README.md: this file
requirements.txt: file needed to install necessary packages
saved_weights.pt: stores the best model from main.py
submission.csv: where the predicted labels are stored in the requested format of ID, label
environment.yml: includes the Miniconda environment to execute this code

# Miniconda Set-up
1. Install Miniconda using: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
2. Replicate the conda environment using: conda env create -f environment.yml
3. Activate the conda evnironment using: conda activate base
4. To deactivate conda, simply run: conda deactivate

# Command Line Instructions
To run the code, simply compile the file using python3 main.py in the command line. This code has been tested to run on python version 3.11.5.
