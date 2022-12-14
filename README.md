# IDD-Papers-AI4DR


This repository allows reproducing the results given the the paper  **AI4DR: Development and Implementation of an Annotation System for High-Throughput Dose-Response Experiments** by Bianciotto et al.

Copyright Notice: Permission is hereby granted, free of charge, for academic research purpose only and for non-commercial use only, to any person from academic research or non-profit organization obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, or merge the Software, subject to the following conditions: this permission notice shall be included in all copies or substantial portions of the Software. All other rights are reserved. The Software is provided 'as is", without warranty of any kind, express or implied, including the warranties of noninfringement. 
An international patent application related to this work has been published under number WO2022/112568. 

A conda environment for making the AI4DR_Tox21_luc_biochem_dataset_analysis.ipynb notebook to run can be installed with the following command :
```
conda create --name AI4DR_env --file AI4DR-env.txt
```

When in this environment, the hillfit library ( https://github.com/himoto/hillfit ) needs to be installed with :
```
pip3 install hillfit
```
The AI4DR_tox21-luc-biochem-p1_CNN_classification.py and AI4DR_tox21-luc-biochem-p1_RF_classification.py scripts allow to compute the classification either with the AI4DR (CNN for shape classification based on DRC images, MLP for dispersion classifier) or with the RF classifier (shape classifiction only).
The AI4DR_Tox21_luc_biochem_dataset_analysis_AI4DR_model.ipynb notebook contains analyses of the AI4DR predictions performed on the Tox21 'tox21-luc-biochem-p1' dataset.
One can find its description at this URL : https://tripod.nih.gov/tox/assays
The data itself can be found at : https://tripod.nih.gov/tox21/pubdata/
A copy of this data is provided in the tox21-luc-biochem-p1 directory for convenience.
The AI4DR_Tox21_luc_biochem_dataset_analysis_RF_model.ipynb notebook contains analyses of the RF classifier performed on the Tox21 'tox21-luc-biochem-p1' dataset.

As the original Tox21 DR data comes from tests on 15 concentrations (CONC0 to CONC14) performed in triplicate (replica 0,1 and 2), and different classifications were performed for each sample using either the full set of DR data or a subset of it. When half the concentrations were considered, the odd concentrations were skipped, leading to build 8 concentrations DR curves. 

The SAMPLE_ID, ASSAY_OUTCOME and CURVE_CLASS2 fields have been left untouched from the input data.

The name of the different AI4DR-related fields in the curves_df dataframe are indicated in the table below :


| Replica considered | Concentrations considered | Concentrations | Raw I percent | I percent after translation | Shape Classification | Shape Probability | Dispersion Classification |  Dispersion Classification Probability | 
|:--------------:|:--------------:|:-------------:|:-------------------:|:----------------:|:----------------------:|:------------------:|:--------------------------:|:--------------------------:|
| Replica 0,1,2  |        all     |  pX_list      |  Y_list_notr        | Y_list           |    category            | probability        |      Disp_model012         |      Disp_Proba012         |
| Replica 0,1    |        all     |  pX_list01    |  Y01_list_notr      | Y01_list         |    category01          | probability01      |      Disp_model01          |      Disp_Proba01          |
| Replica 0,2    |        all     |  pX_list02    |  Y02_list_notr      | Y02_list         |    category02          | probability02      |      Disp_model02          |      Disp_Proba02          |
| Replica 1,2    |        all     |  pX_list12    |  Y12_list_notr      | Y12_list         |    category12          | probability12      |      Disp_model12          |      Disp_Proba12          |
| Replica 0,1,2  |       half     | pXhalf_list   |  Yhalf_list_notr    | Yhalf_list       |    categoryhalf        | probabilityhalf    |      Disp_modelhalf012     |      Disp_Probahalf012     |
| Replica 0,1    |       half     | pXhalf_list01 |  Yhalf01_list_notr  | Yhalf01_list     |    categoryhalf01      | probabilityhalf01  |      Disp_modelhalf01      |      Disp_Probahalf01      |
| Replica 0,2    |       half     | pXhalf_list02 |  Yhalf02_list_notr  | Yhalf02_list     |    categoryhalf02      | probabilityhalf02  |      Disp_modelhalf02      |      Disp_Probahalf02      |
| Replica 1,2    |       half     | pXhalf_list12 |  Yhalf12_list_notr  | Yhalf12_list     |    categoryhalf12      | probabilityhalf12  |      Disp_modelhalf12      |      Disp_Probahalf12      |


The final AI4DR categories and probabilities that are computed in the notebooks have the corresponding variable names:

|  Replica considered | Concentrations considered | Final Category | Final probability |
|:--------------:|:--------------:|:-----------------:|:-------------------:|
| Replica 0,1,2  |        all     |  Final_cat012     |  Y_list_notr        | 
| Replica 0,1    |        all     |  Final_cat01      |  Y01_list_notr      |
| Replica 0,2    |        all     |  Final_cat02      |  Y02_list_notr      |
| Replica 1,2    |        all     |  Final_cat12      |  Y12_list_notr      |
| Replica 0,1,2  |       half     | Final_cathalf012  |  Yhalf_list_notr    |


