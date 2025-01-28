#!/bin/bash

# This is for GuiAgentLoopCore

eval "$(pyenv init -)"
cd GuiAgentLoopCore
pip install --upgrade pip
pip install -e .
