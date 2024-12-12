#!/bin/bash

python -m pip install -r requirements.txt

huggingface-cli login

python demo.py