# NLP Project: Creating a T5 model to detect hate speech/microagressions

This is the code for work being done by Sravani Nanduri, Maarten Sap, and Liwei Jiang.

## Setup Instructions:
All packages can be installed via conda. Install anaconda [here](https://docs.anaconda.com/anaconda/install/index.html),
and then run
```
conda env create -f environment.yml
conda activate yejinProj
```
This installs all packages into a virtual environment called ```yejinProj```.
Make sure to activate it to run this code.

In order to run the workflow, run
```
snakemake --cores 4
```
where the Snakefile is located. This should run the model with the data, the predictions, and the accuracy metrics.
Make sure to change the "output" in the snakefile to change where the information saves.

Please email [nandsra@uw.edu](mailto:nandsra@uw.edu) with any questions.
