#!bin/bash

defectguard \
    -models deepjit \
    -dataset platform \
    -repo /home/manh/Documents/DefectGuard/Tic-tac-toe-Game-using-Network-Socket-APIs \
    -uncommit \
    -top 10 \
    -sort \
    -main_language C
