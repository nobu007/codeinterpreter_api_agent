#!/bin/bash

# This is for GuiAgentLoopCore

eval "$(pyenv init -)"
cd langchain-google/libs/genai
pip install -e .
