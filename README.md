# Prediting peptide binding sites using support vector machine

### [1]. How to Run Package:

#### [2.1] Create conda environment with all required packages
```console
user@machine:~$ conda create --name env_name --file requirements.txt
```
#### [2.2] Activate the environment
```console
user@machine:~$ conda activate env_name
```
#### [2.3] Test Command-line #1: Run on Default Dataset
```console
user@machine:~$ python main.py
```
#### [2.4] Test Command-line #2: Run on different dataset
```console
user@machine:~$ python main.py -train train_dataset.csv -test test_dataset.csv -f 'MonoOccur' -m 'svm-linear'
```

**Table 1:**  command line element
| Symbol  | Explanation  |
| ------- | ------------ |
| -train| Train dataset path  |
| -test | Test dataset path |
| -f | Feature name |
| -m | Model name