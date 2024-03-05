# stilsucher
Fashion Semantic Search Engine

## Environment Setup

First of all make sure you have Docker installed in your machine, since it is required to run the Qdrant and the Application.

Then Create a new virtual environment using the following command (or the equivalent depending on what solution you are using to create virtual environments):
```bash
python3 -m venv venv
```

### Using the setup script

After creating the virtul environment, activate it and run the following command to install and setup everything for the notebooks and the application to work
```bash
bash setup.sh
```

This will install the required packages, build the application docker image and download the dataset.

### Doing it manually step by step

Build the application docker image using theollowing command:
```bash
``` in the `requirements.txt`file using the following command:
```bashbash
docker build -t stilsucher .
``` 

Install the required packages in th```
e `requirements.txt`file using the following command:
```bash
pip install -r requirements.txt
```

Unzip the `data.zip` file. This contains an already initialized Qdrant index with the fashion dataset and will be used as a volume for the Qdrant container.
```bash
unzip data.zip
```

The notebooks have cells to download the dataset that contains all the images, but in case you want to run the application directly, you can download it running the following command:
```bash
cd experiments
gdown "1igAuIEW_4h_51BG1o05WS0Q0-Cp17_-t&confirm=t"
unzip data.zip
```
