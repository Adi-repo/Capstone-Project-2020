```
Model - P 1-D CNN
Accuracy - 97.0% 
Reference - https://arxiv.org/pdf/1608.04064v1.pdf 

```
- ##### The Dataset is downloaded from Temple University EEG Corpus. It consists of data of 1000(500 normal & 500 abnormal) people in which data of 800(400 normal & 400 abnormal) people is used for training and data 200(100 normal & 100 abnormal) people for testing.

> Here normal refers to people who are not suffering from epilepsy and abnormal refers to people who are suffering from epilepsy. 
- ##### The Code is written in Python3. To check the results follow below given steps:

  1. To install required packages run command -
   > pip3 install -r requirements.txt
  2. To check results run command -
   > python3 evaluate.py
