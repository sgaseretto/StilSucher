#!/bin/bash

# Build the application docker image
docker build -t stilsucher .

# Install the required packages
pip install -r requirements.txt

# Unzip the data.zip file
unzip data.zip

# Download the dataset that contains all the images if it doesn't exist
cd experiments
if [ ! -f data.zip ]; then
    gdown "1igAuIEW_4h_51BG1o05WS0Q0-Cp17_-t&confirm=t"
fi
unzip data.zip

# Navigate back to the root directory
cd ..