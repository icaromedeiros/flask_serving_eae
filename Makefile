###
# Makefiles
### 

# A Makefile is a multi-language standard for declaring the most important commands in a software project lifecyle
# It's a good practice to try and understand a software project source code to start by reading Makefiles

# It usually includes:
#   - Build/compile commands
#   - Commands to run your software
#   - Run test suites and other quality control features
###

# Creates a virtual env to isolate your environment from other Python projects
venv:
	python -m venv .venv
	echo "Check if .venv directory was created"
	echo "Remember to use source .venv/Scripts/activate in Windows or .venv/bin/activate in MacOS/Linux"
# Install main requirements (for the Flask Server API)
install:
	pip install -r requirements.txt
# Install requirements for model training / evaluation scripts
install_experiments:
	pip install -r experiments/requirements.txt