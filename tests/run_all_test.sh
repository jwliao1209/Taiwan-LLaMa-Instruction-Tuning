#!/bin/bash

bash run.sh "${1}" "${2}" "${3}" "${4}"
bash tests/test_ppl.sh "${1}" "${2}" "${3}"
