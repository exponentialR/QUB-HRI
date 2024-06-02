#!/bin/bash

# Define variables
DIRECTORY="/home/iamshri/ml_projects/Datasets/QUB-PHEO/"
START_PARTICIPANT=1
END_PARTICIPANT=3
EMAIL="samueladebayo@ieee.org"
FROM_EMAIL="sadebayo01@qub.ac.uk"

# Function to send email
send_email() {
    local subject=$1
    local body=$2
    echo -e "To: $EMAIL\nFrom: $FROM_EMAIL\nSubject: $subject\n\n$body" | ssmtp $EMAIL
}
ok
# Send start notification email
send_email "Zipping Process Started" "The zipping process has started."

# Run the Python script
python3 zip_folders.py "$DIRECTORY" "$START_PARTICIPANT" "$END_PARTICIPANT"

# Send completion notification email
send_email "Zipping Process Completed" "The zipping process has completed."
