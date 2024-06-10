#!/bin/bash

# Change directory to the specified path
cd /home/iamshri/PycharmProjects/QUB-HRI/actionLabelling

# Activate the virtual environment
source action-label/bin/activate


label-studio & sleep 10; disown
