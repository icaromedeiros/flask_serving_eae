# Recheck everything and prioritize installation stuff

- https://github.com/icaromedeiros/flask_serving_eae (Fork and clone)
  - I'm going to need a volunteer with Python, VSCode and git working locally to create a fork, a clone and a Pull Request
  - **PR IDEA** Test versions other than Python 3.10
  - I'll be answering [Issues](https://github.com/icaromedeiros/flask_serving_eae/issues), just in case
- Virtual environments
  - Install /requirements.txt (remove version)
  - `nice-to-have` Install `/experiments/requirements.txt` meanwhile `Go to next point`
- Use decision tree classifier to export pickle (in `/experiments`)
  - Run experiments as scripts vs notebooks (show two versions)
  - [**TODO** `nice-to-have`] internal outputs like the decision tree itself,
    besides the predict method with `?explain=true`
- data/ is in .gitignore
  - add to git afterwards so everyone can fork
- Flask crash course
- Run local server
- Tests (in `__main__`), pytest
- **TODO** `nice-to-have` Now we want to create a `/flower/<id>` route