# SurvivalAnalysisCardiology

This repository contains the code needed to perform the experiments described in "Linear and Machine Learning-based methods for survival analysis in patients with chronic heart failure" by Alicia Olivares-Gil, José A. Pérez-Rivera, Óscar Hernández-Santos, Juan J. Rodríguez and José F. Díez-Pastor. 

It is a study that evaluates statistical and machine learning methods for survival analysis in heart failure (HF) patients, with clinical data from 553 patients, including 76 features per patient, and with time to first HF-related hospitalization as the primary endpoint. 

## Requirements
### Data
Notebook `Toy_data_generation.ipynb` generates a toy dataset with the same structure as the one used in the study so the code in this repository can be tested. The actual data used in our study are available from the corresponding author upon reasonable request.

### Conda environment 
All the Python code was executed using the conda environment available in this repository. 
To install this conda environment: 
```
conda env create -f survcardio.yml
```
To activate the environment: 
```
conda activate survcardio
```
## Usage
In order to reproduce the results shown in the paper, follow these steps: 
### 1. Calculate Kaplan-Meier curves 
The Kaplan-Meier curves for all individual features are shown in [Kaplan-Meier_curves.ipynb](https://github.com/aliciaolivaresgil/SurvivalAnalysisCardiology/blob/main/Kaplan-Meier_curves.ipynb).
### 2. Perform experiments 
```
python Comparison_Cox.py
python Comparison_ML.py
```
### 3. Visualize results
The performance of each of the methods is shown in [Results.ipynb](https://github.com/aliciaolivaresgil/SurvivalAnalysisCardiology/blob/main/Results.ipynb).
### 4. Calculate best features
The most predictive features for each model are calculated in [Best_features.ipynb](https://github.com/aliciaolivaresgil/SurvivalAnalysisCardiology/blob/main/Best_features.ipynb).
### 5. Perform statistical tests 
Results for the Nemenyi test are calculated and shown in [Statistical_tests.ipynb](https://github.com/aliciaolivaresgil/SurvivalAnalysisCardiology/blob/main/Statistical_tests.ipynb). 
