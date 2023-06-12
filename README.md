# Just-in-time Defect Prediction Tool

## About

* A tool that can predict wether the commit is defect (has bug) or not

## Technologies

* Using `Flask` to define API
* Using `pytorch` for Deep Learning
* Using `sklearn` for Machine Learning 

## Prerequisite

* `requirements.txt` in each services

## How to run

### To run all services
```
cd [service]
python [service].py
```

### To use tool
Using `help` for details
```
python main.py --help
```

Example: check if the HEAD commit in a git repository is defect or not using CC2Vec and DeepJIT
```
python main.py \
    -deep cc2vec deepjit \
    -repo "Tic-tac-toe-Game-using-Network-Socket-APIs" \
    -commit_hash HEAD
``` 
  
## Documents
* My Notion: [link](https://tree-makemake-699.notion.site/Just-in-time-Defect-Prediction-Tool-c9d4892ceca84b4ab6dcf1f0574b1355?pvs=4)
