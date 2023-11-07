#!bin/bash

defectguard \
    -models deepjit \
    -train_data platform \
    -hyperparams hyperparams.json \
    -train
