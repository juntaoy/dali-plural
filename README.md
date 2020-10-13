# Free the Plural: Unrestricted Split-Antecedent Anaphora Resolution

## Introduction
This repository contains code introduced in the following paper:
 
**[Free the Plural: Unrestricted Split-Antecedent Anaphora Resolution]()**  
Juntao Yu, Nafise Sadat Moosavi, Silviu Paun and Massimo Poesio  
In *Proceedings of he 28th International Conference on Computational Linguistics (COLING)*, 2020

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that modify and run `extract_bert_features/extract_bert_features.sh` to compute the BERT embeddings for your training or testing.
* You also need to download context-independent word embeddings such as fasttext or GloVe embeddings that required by the system.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://www.dropbox.com/s/zn4nrqqn07ats23/coling2020%20plural%20best%20model.zip?dl=0). We provide the best model for reported in our paper.
* Choose the model you want to use and copy them to the `logs/` folder.
* Modifiy the *test_path* accordingly in the `experiments.conf`:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file is a batch of sentences and must in the following format:
   
   ```
   {
  "clusters": [[0, 4],[1],[2],[3],], #Coreference use the indices of the mention
  "mentions": [[0,0],[2,3],[5,5],[7,8],[10,10],[12,13]], #mentions [start_index, end_index]
  "plurals": [[5,1],[5,3]], #plural [anaphor, antecedent] pairs, "both car" --> "a car", "another car"
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["Mary", "has", "another", "car", "."] ["John", "washed", "both", "car", "yesteday","."]]
  }
  ```
  
  * The mentions only contain two properties \[start_index, end_index\] the indices are counted in document level and both inclusive.
  * For coreference clusters (includes singleton clusters) are represented by their mention indices.
  * For plural pairs, each pair contains two mention indices the first one is the anaphora and the second one is the antecedent.
* Then use `python evaluate.py config_name` to start your evaluation

## To train your own model
* You will need additionally to create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* Then you can start training by using `python train.py config_name`
