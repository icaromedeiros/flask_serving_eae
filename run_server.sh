#!/bin/bash

###
# Dependencies
###

# - Windows: ALWAYS run in git bash, Powershell will not run this
# - MacOS, Linux: Probably any terminal will do

###
# ERRORS
###

#zsh: permission denied: ./run_server.sh
# If you see messages like this, run the following command: chmod +x run_server.py

###
# One-liner (use it instead of this script if there are errors)
###

# export FLASK_APP=iris_api && export FLASK_DEBUG=1 && flask run --port=9000

###
# Environment variables
###

# Name of the main module iris_api.py
export FLASK_APP=iris_api
# Enables automatically realoading
export FLASK_DEBUG=1

# By default, 5000 is the port used, in MacOS 5000 is used for AirPlay
# Check your OS security constraints
flask run --port=9000

# If you want to broadcast your server (i.e. make it available ion local network),
#   use the --host command
# However, some networks restrict these kind of broadcast for security reasons,
#   contact your network administrator
#flask run --host=0.0.0.0