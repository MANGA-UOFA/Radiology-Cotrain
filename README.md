This is the official repository for the paper "A Dual View Approach to Radiology Report Analysis by Co-Training" published in LREC-COLING 2024

** Data is not available due to confidentiality issues with patient data ** 

## Setup 
```
conda create -n cotrain
conda activate cotrain
conda install --file requirements.txt
```

## Data Processing
Since the data is private, experiments from the paper cannot be reproduced. However, you can use your own data to apply our method to your own data. The data-processing procedure we used can be found in the folder `data_proc/`. 

## Co-train and Ensemble:
To use our method on your own data, run

```
python co-train.py \
[--logdir <WHERE TO STORE RESULTS AND CHECKPOINTS>] \
[--labeled-pickle <PICKLE CONTAINING LABELED DATA DATAFRAME>]
[--unlabeled-pickle <PICKLE CONTAINING UNLABELED DATA DATAFRAME>]
[--model-name <NAME OF MODEL>]
[--view1-name <COLNAME OF VIEW1>]
[--view2-name <COLNAME OF VIEW2>]
[--target <COLNAME OF TARGET>]
...
```
**Dataframe format**
The dataframe that is contained in the pickle file must have the following columns:
1. **view1:** The text corresponding to the first view
1. **view2:** The text corresponding to the second view
1. **target:** The label corresponding to the two views
