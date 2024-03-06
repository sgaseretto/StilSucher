#!/bin/bash

# Set the working directory to /app
cd /app

# Check if the required files and folders exist in /app
if [ -d "assets" ] && [ -f "rxconfig.py" ]; then
    # Check if the .reflex directory exists in $HOME
    if [ -d "$HOME/.reflex" ]; then
        # .reflex exists, run the migrate and run commands
        reflex db migrate && reflex run
    else
        # .reflex doesn't exist, initialize reflex and then migrate and run
        reflex init
        reflex db migrate && reflex run
    fi
else
    # assets and rxconfig.py don't exist, run the init command
    reflex init
    reflex db init
    reflex run
fi
