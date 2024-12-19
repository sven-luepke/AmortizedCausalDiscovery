#!/bin/bash
python -m data.generate_dataset
python -m train --suffix _springs5  # observed
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 0  # ACD with latent
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 1  # None
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 2  # Mean