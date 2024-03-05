#!/bin/bash

# Build the application docker image
docker build -t stilsucher .
echo "App docker image built"

# Install the required packages
pip install -r requirements.txt

# Unzip the data.zip file
unzip qdata.zip
rm -rf data/.DS_Store
rm -rf data/collections/fclip/0/segments/.DS_Store
echo "qdata.zip unzipped"

# Download the dataset that contains all the images if it doesn't exist
if [ ! -f "data.zip" ]; then
    echo "Downloading data.zip"
    gdown "1igAuIEW_4h_51BG1o05WS0Q0-Cp17_-t&confirm=t"
    unzip data.zip
    echo "data.zip downloaded and unzipped"
fi

# check if there already is a folder called data_for_fashion_clip in the experiments folder
if [ ! -d "experiments/data_for_fashion_clip" ]; then
    echo "Creating data_for_fashion_clip folder in experiments"
    mv data_for_fashion_clip experiments/
    echo "data_for_fashion_clip moved to experiments"
fi

if [ ! -d "app/assets/data_for_fashion_clip" ]; then
    echo "Moving data_for_fashion_clip folder in app/assets"
    mv data_for_fashion_clip app/assets/
    echo "data_for_fashion_clip moved to app/assets"
fi