#!/bin/bash
# Make sure you run conda activate <env> first
# to run this from an environment where the allenSDK is installed

python deploy_glm_fits.py --version 102_active_mongo --env-path /home/alex.piet/codebase/miniconda3/envs/np --src-path /home/alex.piet/codebase/NP/visual_behavior_glm_NP/ --job-end-fraction 1 
