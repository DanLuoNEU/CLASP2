#!/bin/bash
# conda init bash
# conda activate dyanEnv
cd ./AlphaPose/
python demo.py --indir $1 --outdir $2 --fast_inference False
cd ..
# conda activate clasp_i3d_demo
