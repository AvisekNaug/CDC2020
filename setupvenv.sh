#!/bin/bash
#--script to quivcly install all the necessary requirements

# install python3-dev if not present
sudo apt-get install python3-dev

# create venv
python3 -m venv cdc2020

#--activate the venv
source cdc2020/bin/activate

# upgrade pip
pip install --upgrade pip

#--Install all requirements
pip install -r minimalrequirements.txt

#--Install jupyter notebook extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user

# add notebook to IPython kernel environment
python -m ipykernel install --user --name=cdc2020
