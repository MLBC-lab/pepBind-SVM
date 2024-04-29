# Prediting peptide binding sites using a support vector machine

### [1]. How to Run Package:

#### [1.1] Create a conda environment with all required packages
```console
user@machine:~$ conda create --name env_name --file requirements.txt
```
#### [1.2] Activate the environment
```console
user@machine:~$ conda activate env_name
```
#### [1.3] Test Command-line #1: Run on Default Dataset
```console
user@machine:~$ python main.py
```
#### [1.4] Test Command-line #2: Run on different dataset
```console
user@machine:~$ python main.py -train train_dataset.csv -test test_dataset.csv -f ['MonogramOccur'] -m 'svm-linear'
```

**Table 1:**  command line element
| Symbol  | Explanation  |
| ------- | ------------ |
| -train| Train dataset path  |
| -test | Test dataset path |
| -f | List of features separated by commas: MonogramComp, BigramComp, MonogramOccur, BigramOccur|
| -m | Model name (svm-rbf/svm-linear)
