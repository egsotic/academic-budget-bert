#! /bin/bash

docker run -tid \
    --gpus 4 \
    --name deepspeed \
    -v /home/nlp/egsotic/repo/academic-budget-bert/:/project \
    -v /dev/shm:/dev/shm \
    -v /home/nlp/egsotic:/home/nlp/egsotic \
    deepspeed_10_5_23