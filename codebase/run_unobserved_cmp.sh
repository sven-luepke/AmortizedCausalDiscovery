#!/bin/bash

# old model - new model v1 - new model v2
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=8 --unobserved 1 --model_unobserved 3 --predict_initial_point
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=8 --unobserved 1 --model_unobserved 3
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=8 --unobserved 1 --model_unobserved 0