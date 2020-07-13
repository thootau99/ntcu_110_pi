from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify

import requests
import os
app = Flask(__name__)
ip = "25.16.141.60:5000"
@app.route('/takeoff')
def takeoff():
    url = "http://" + ip + "/takeoff"
    requests.get(url)
    return "0"
@app.route('/land')
def land():
    url = "http://" + ip + "/land"
    requests.get(url)
    os.system("echo land success?")
    return "0"

app.run(host="0.0.0.0")
