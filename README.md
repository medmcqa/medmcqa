<h1 align="center">MedMCQA </h1>

<h3 align="center">MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering</h3>

A large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address realworld medical entrance exam questions. 

The MedMCQA task can be formulated as X = {Q, O} where Q represents the questions in the text, O represents the candidate options, multiple candidate answers are given for each question O = {O1, O2, ..., On}. The goal is to select the single or multiple answers from the option set.


[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub commit](https://img.shields.io/github/last-commit/medmcqa/medmcqa)](https://github.com/medmcqa/medmcqa/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


## Requirements

`pip3 install -r requirements.txt`


## Data Download and Preprocessing

download the data from below link

data : https://drive.google.com/uc?export=download&confirm=t&id=1MtcH29lUnaPehvUi2tZ9KEdgZy-NgOH2

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


## Experiments code

To run the experiments mentioned in the paper, follow the below steps
- Clone the repo
- Install the dependencies 

`pip3 install -r requirements.txt`

- Download the data from google drive link
- Unzip the data
- run below command with the data path

` python3 train.py --model bert-base-uncased --dataset_folder_name "/content/medmcqa_data/" `


## Model Submission and Test Set Evaluation

To preserve the integrity of test results, we do not release the test set's ground-truth to the public. Instead, we require you to
use the test-set to evaluate the model and send the predictions along with unique question id `id` in csv format

Example :

| id    | Prediction (correct option)  | 
| :-------------: |:-------------:|
| 84f328d3-fca4-422d-8fb2-19d55eb31503 |  2  | 
| bb85e248-b2e9-48e8-a887-67c1aff15b6d |  3  | 

aadityaura [at] gmail.com,
logesh.umapathi [at] saama.com
