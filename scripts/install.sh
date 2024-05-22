#!/bin/bash


WORKING_DIR=$(pwd)
USE_DEV=0

# ask if working directory is correct
echo "Current working directory is $WORKING_DIR"
read -p "Is this the correct working directory? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Please run this script in the correct working directory"
    exit 1
fi

# make venv
python3 -m venv venv

# activate venv
source venv/bin/activate


# install requirements
if [ $USE_DEV -eq 1 ]; then
    pip install -e .[dev]
    pre-commit install
else
    pip install -e .
fi


