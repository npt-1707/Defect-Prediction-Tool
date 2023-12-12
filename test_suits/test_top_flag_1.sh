#!bin/bash

defectguard \
    -models deepjit \
    -dataset platform \
    -repo /home/manh/Documents/DefectGuard/Tic-tac-toe-Game-using-Network-Socket-APIs \
    -commit_hash uncommit dedbc4ed5b953ac03854c83e0b91dd56f4fd1f1e e07e944afd5435a367c8b1789cda20d7c52240c3 c02ebd96206ea49a172c49cf6ee47ce6e9b7cea6 5b55c88d5809bb75db413247eeb7ab9aac94ea46 \
    -top 10 \
    -main_language C
