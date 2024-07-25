#!/bin/bash

# Define constants
OUTPUT_DIR="/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/TRIANGULATED"
CALIB_DIR="/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/CALIBRATION"
SCRIPT_PATH="/home/samueladebayo/PycharmProjects/QUB-HRI/reconstruction/triangulate_views.py"
VENV_PATH="/home/samueladebayo/PycharmProjects/QUB-HRI/pheo"  # Path to the virtual environment

# Activate the Python virtual environment
source "${VENV_PATH}/bin/activate"


# Subtask directories and their total file counts
declare -A subtasks=( ["HHO"]=2065 ["HCP"]=120 ["FO"]=70 ["HPT"]=2710 ["CS"]=1980 ["RHO"]=3865 ["HCS"]=435 ["HBPP"]=1320 ["RCG"]=940 ["RPT"]=2955 ["RPS"]=4715 ["RBPS"]=985 ["REC"]=40 ["STP"]=3880 ["HSA"]=720 ["HBNS"]=1125 ["HEC"]=75 ["RNS"]=1140 ["HGOT"]=2290 ["RSA"]=670 ["HPP"]=1815 ["THP"]=455 ["HPCS"]=1845 ["HBPS"]=1100 ["RBNS"]=1065 ["HPH"]=730 ["CG"]=620 ["HNS"]=3360 ["RBPP"]=1425 ["HPS"]=4820 ["RPH"]=695  ["RPCS"]=35 ["RCS"]=725 ["RPP"]=1715)

# Process each subtask
for subtask in "${!subtasks[@]}"; do
    # Calculate the number of files to process by dividing by 5
    total_files=${subtasks[$subtask]}
    let processed_files=total_files/5

    # Calculate number of batches (100 files per batch)
    let batches=(processed_files+99)/100  # Adding 99 ensures rounding up

    # Define subtask directory (assuming a specific structure, update as necessary)
    SUBTASK_DIR="/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/CAM_LR/${subtask}"

    echo "Processing subtask: $subtask"
    echo "Total batches to process: $batches"

    # Run the Python script for each batch
    for (( batch=1; batch<=batches; batch++ )); do
        echo "Running batch $batch/$batches for subtask $subtask"
        python $SCRIPT_PATH --subtask_dir "$SUBTASK_DIR" --output_dir "$OUTPUT_DIR" --calib_dir "$CALIB_DIR"
    done
done
deactivate


