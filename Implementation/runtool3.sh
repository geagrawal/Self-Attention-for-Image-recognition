#!/bin/bash

# This is where the benchmark log is going to reside.
# target=/s/kabdulma/Workspace/experiment_raport.json
echo "Saving results to: $target"
echo "Test running on: $(hostname)"
cd /s/kabdulma/Workspace/SAN-master/tool;
nvidia-smi;
python --version;
python -c 'import torch; print(torch.__version__)';
cd /s/kabdulma/Workspace/SAN-master/tool/;
export PYTHONPATH="/s/kabdulma/Workspace/SAN-master/:$PYTHONPATH";

# python san.py;
python newrun.py patch;
echo "Completed the Patchwise Run and curve png saved"
python newrun.py pair; 
echo "Completed the Pairwise Run and curve png saved"

#cp /s/kabdulma/Workspace/CyclicGans/Allfiles/Allfiles/experiment_raport.json $target