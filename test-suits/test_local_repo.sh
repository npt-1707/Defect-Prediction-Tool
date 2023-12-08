#!bin/bash

defectguard \
    -models deepjit \
    -dataset platform \
    -repo . \
    -commit_hash 9cd64889990fd91e0396495f622e536d02e7bf88 061b5c1daf3363f5a88493cba4c6f7e06c934fea \
    -main_language Python
