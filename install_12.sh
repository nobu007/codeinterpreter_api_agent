#!/bin/bash

# This is for pygraphviz
sudo apt install -y python3-dev graphviz libgraphviz-dev pkg-config
sudo pip install pygraphviz

# This is for zoltraak_auto
eval "$(pyenv init -)"
cd zoltraak_auto
pip install -e .
