# Standard Medical Imaging Noninvasively Predicts Nanomedicine Tumor Accumulation Using Interpretable Radiomics
<img width="1098" alt="fig1" src="https://github.com/user-attachments/assets/ce6179e1-2ea7-4b3a-983e-97ba127f6fb5">

The repository serves as the official implementation of the paper titled "Standard Medical Imaging Noninvasively Predicts Nanomedicine Tumor Accumulation Using Interpretable Radiomics". This prediction is characterized by offering a non-invasive and effective method to aid in patient stratification based on tumour nanomedicine accumulation. This study underscores the potential of utilizing radiomics models, which integrate standard-of-care medical imaging with artificial intelligence, to accurately predict nanomedicine accumulation within tumours. The procedure is illustrated by the figure above. 
# Environment Setup
The code has been successfully tested using Python 3.11. Therefore, we suggest using this version or a later version of Python. A typical process for installing the package dependencies involves creating a new Python virtual environment.
# Reproducing experiments
To reproduce the experiments illustrated in the paper, one can use pora.py script. One needs to change the 3 paths in the script (“input_path”, “output_path”, and “config_path”) to the paths corresponding to your own files. In detail, the path of the folder containing the data for all the feature variables (e.g., cancer type, GNP sizes, CEUS-derived parameters, SWE mean, and CT and US radscores data) needs to be assigned to the “input_path” variable. The folder path of the filtered radiomic features data, and the radscores of these features, as well as the predicted effects of the machine learning models, need to be assigned to the “output_path” variable. In addition, the path of the config folder we uploaded needs to be assigned to the “config_path” variable, i.e.
```
if __name__ == "__main__":
    input_path = "/dataflow/DS02-Radiomics/input/"
    output_path = "/dataflow/DS02-Radiomics/output/"
    config_path = "/dataflow/DS02-Radiomics/config/"
    section = "train"
    if section == "train":
        train(input_path, output_path, config_path)
    elif section == "infer":
        infer(input_path, output_path, config_path)
```
