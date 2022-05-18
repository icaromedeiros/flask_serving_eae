# flask_serving_eae

Simple API to serve dataset info and a ML model based on the Scikit-learn iris dataset

# Goal

**TODO** Define business goals

# Dependencies

- Tested in Python 3.10 (should work with Python 3.6+, if not, open an [issue](https://github.com/icaromedeiros/flask_serving_eae/issues/new))
  - Students can add their own versions in which it was tested, create a Pull Request for this README.md)
- Virtualenv:
  - python -m venv .venv (check if the dir exists)
  - Non-Windows:  `source .venv/bin/activate` (you're going to see `(.venv)` in your prompt
  - Windows: `source .venv/Scripts/activate`
- Use `make install` or directly `pip install -r requirements.txt`

# Running

**REPRODUCIBILITY MANTRA**: help your colleagues

- Create executable permissions to the shell script `chmod +x run_server.sh`
- Run with the following alternatives:
  - As Python module `python iris_api.py`
  - Flask local server oneliner `export FLASK_APP=iris_api && export FLASK_DEBUG=1 && flask run --port=9000`
  - `sh run_server.sh`
