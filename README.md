# Interpretable Multimodality-Radiomics-Based Machine Learning Model Predicts the Tumour Accumulation of Nanomedicines
![whole](https://github.com/user-attachments/assets/0e51b252-9da7-48b0-883a-8036f4ea8f2e)
The repository serves as the official implementation of the paper titled "Interpretable Multimodality-Radiomics-Based Machine Learning Model Predicts the Tumour Accumulation of Nanomedicines". This prediction is characterized by offering a non-invasive and effective method to aid in patient stratification based on tumour nanomedicine accumulation. This study underscores the potential of utilizing radiomics models, which integrate standard-of-care medical imaging with artificial intelligence, to accurately predict nanomedicine accumulation within tumours. The procedure is illustrated by the figure above. 
# Environment Setup
The code has been successfully tested using Python 3.11. Therefore, we suggest using this version or a later version of Python. A typical process for installing the package dependencies involves creating a new Python virtual environment.
# Reproducing experiments
To reproduce the experiments illustrated in the paper, one can use pora.py script, which takes three main arguments to specify the machine learning model, i.e.
```
if __name__ == "__main__":
    input_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/input/internal/"
    output_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/output/"
    config_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/config/"
    section = "train"
    if section == "train":
        train(input_path, output_path, config_path)
    elif section == "infer":
        infer(input_path, output_path, config_path)
```
