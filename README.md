# flask_serving_eae

Simple API to serve dataset info and a ML model based on the Scikit-learn iris dataset

# Goal

Here you can define business goals

# Dependencies

- Tested in Python 3.10 (should work with Python 3.6+, if not, open an [issue](https://github.com/icaromedeiros/flask_serving_eae/issues/new))
  - Students can add their own versions in which it was tested, create a Pull Request for this README.md)
- (Optional) Create a virtualenv if you don't want to rewrite your environment with new library versions instead of the ones from Anaconda
- Virtualenv:
  - python -m venv .venv (check if the dir exists)
  - Non-Windows:  `source .venv/bin/activate` (you're going to see `(.venv)` in your prompt
  - Windows: `source .venv/Scripts/activate`
- Use `make install` or directly `pip install -r requirements.txt`

# Running

It's considered good practice to include running instructions to help other **reproduce** (run, test and evolve) your software.

Run the Flask Server using the following alternatives:

1. As a Python module: `python iris_api.py` (check the end of the file: the `__main__` part)
1. Flask local server oneliner: `export FLASK_APP=iris_api && export FLASK_DEBUG=1 && flask run --port=9000`
  1. Here you're defining environment variables and arguments like port to `flask` binary to know how to execute and which module `iris_api` to run
1. Finally, you can run it as a script in `sh run_server.sh`
  1. (If you see permission denied errors) Give executable permissions to the shell script using `chmod +x run_server.sh`
  1. Then, run `sh run_server.sh`

# API Examples

Run these examples in your browser to test main features of your Web Server

- `http://<your_host>:<your_port>/api/iris/classify?sl=5.1&sw=3.5&pl=1.4&pw=0.2`
- `http://localhost:9000/api/iris/classify?sl=5.1&sw=3.5&pl=1.4&pw=0.2` 
- Create executable permissions to the shell script `chmod +x run_server.sh`
- Run with the following alternatives:
  - As Python module `python iris_api.py`
  - Flask local server oneliner `export FLASK_APP=iris_api && export FLASK_DEBUG=1 && flask run --port=9000`
  - `sh run_server.sh`
