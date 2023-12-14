#!bin/bash

defectguard \
    -models deepjit \
    -dataset platform \
    -repo /home/manh/Documents/DefectGuard/Tic-tac-toe-Game-using-Network-Socket-APIs \
    -uncommit \
    -top 10 \
    -main_language C \
    -debug \
    -log_to_file
