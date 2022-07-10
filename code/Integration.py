# Runs python file and displays the output on the webpage
# Under the assumption that the file is in the same directory as this main.py file

import Backend
from asyncio.windows_events import NULL
from pickle import TRUE
from flask import Flask, render_template, request, redirect
import csv
from numpy import True_
app = Flask(__name__)

@app.route('/Result/', methods=['get', 'post'])
def result():
    message='Sumitted Successfully'
    if Backend.heartdisease() == 0: 
        message = 'You are not at risk for heart disease.' #after algo integration, add another if else statement here for at risk/not at risk!
    elif Backend.heartdisease() == 1:
        message = 'You are at risk for heart disease.'
    
    if request.method == 'POST':
        return redirect(f'/') #go to homepage
    else: 
        return render_template('Result.html', message=message)