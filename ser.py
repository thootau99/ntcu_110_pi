from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify

import os
app = Flask(__name__)

@app.route('/takeoff')
def takeoff():
    os.system("echo takeoff success")
    return "0"
@app.route('/land')
def land():
    os.system("echo land success?")
    return "0"


app.run(host="0.0.0.0")