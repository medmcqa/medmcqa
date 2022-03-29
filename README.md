<h1 align="center">MedMCQA </h1>

<h3 align="center">MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering</h3>

A large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address realworld medical entrance exam questions. 

The MedMCQA task can be formulated as X = {Q, O} where Q represents the questions in the text, O represents the candidate options, multiple candidate answers are given for each question O = {O1, O2, ..., On}. The goal is to select the single or multiple answers from the option set.


[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub commit](https://img.shields.io/github/last-commit/medmcqa/medmcqa)](https://github.com/medmcqa/medmcqa/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


## Requirements

Python 3, pytorch
To install pytorch , follow the instructions at https://pytorch.org/get-started/


## Data Download and Preprocessing

download the data from below link

data : https://drive.google.com/file/d/1-5Zl3QKNC5OuZpF8RzciJJKFq3Ug2wnT/view?usp=sharing

## Data Format

The structure of each cav file :

- `id`           : Unique id for each sample
- `question`     : Medical question (a string)
- `opa`          : Option A 
- `opb`          : Option B
- `opc`          : Option C
- `opd`          : Option D
- `cop`          : Correct option (Answer option of the question)
- `choice_type`  : Question is single choice or multi-choice
- `exp`          : Explanation of the answer
- `subject_name` : Medical Subject name of the particular question
- `topic_name`   : Medical Subject's topic name of the particular question


## Model Submission and Test Set Evaluation

To preserve the integrity of test results, we do not release the test set's ground-truth to the public. Instead, we require you to
use the test-set to evaluate the model and send the predictions along with unique question id `id` in csv format

Example :

| id    | Prediction (correct option)  | 
| :-------------: |:-------------:|
| 1ert8213 |  0  | 
| 232ee124 |  3  | 

aadityaura [at] gmail.com,
logesh.umapathi [at] saama.com
