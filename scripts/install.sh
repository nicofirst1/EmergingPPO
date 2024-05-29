#!/bin/bash


WORKING_DIR=$(pwd)

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
pip install -e .


