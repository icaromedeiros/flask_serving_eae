'''
Based on https://flask.palletsprojects.com/en/2.1.x/quickstart/#a-minimal-application
'''
# FIXME PEP8 import order: stdlib, reqs, your own modules
import pandas
from flask import Flask, request, jsonify

# __name__ is a special builtin in Python to get:
#  the name of the file you're running
#  or __main__ if you calling Python from the CLI
# Uncomment the following line and test running python app.py
#print(__name__)
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "This is the version 1.x of my API and this route can be used as healthcheck"


@app.route('/api/iris/classify', methods=["GET"])
# FIXME add a POST using request.method == 'POST' and request.json
# TODO [nice-to-have] define content type header, accept JSON, 400 errors 
# TODO [nice-to-have] defensive programming and 500 errors
# TODO [nice-to-have] Include features in JSON result
def classify():
    '''
    This method calls classifier module to get the 
    '''
    # Gets query or request parameters, i.e. ?name=value&name2=value2 in the URL
    # **TODO** What is the primitive type of a query parameter from an URL ? 
    content = request.args

    # TODO create feature vector from JSON
    # TODO call classifier module and discuss about cohesion/coupling SE concepts

    return content
    # return jsonify({"species": "setosa"})


if __name__ == "__main__":
    app.run(port=9000, debug=True)
    # TODO nice-to-have sys args 
    #app.run()