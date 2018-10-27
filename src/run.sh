#!/bin/bash
python create_training_set_N1.py
python create_training_set_N2.py
python create_training_set_N3.py
python create_training_set.py
python mk_submission.py
