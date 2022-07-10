# Main webpage demo code.  Handle form data from website and call the algorithm.
# Author: Victoria Hartman, initially adapted from https://developer.mozilla.org/en-US/docs/Learn/Forms/Sending_and_retrieving_form_data 
# and https://overiq.com/flask-101/form-handling-in-flask/
# Last Updated July 2022

from asyncio.windows_events import NULL
from pickle import TRUE
from flask import Flask, render_template, request, redirect
import csv

from numpy import True_
app = Flask(__name__) #required to setup the dev server aka local host

@app.route('/', methods=['get', 'post'])  #app route is the browser address, / is default.
def index():
    if request.method == 'POST':
        return redirect(f'/Consent/') #go to the consent page   
    
    return render_template('index.html', name='homepage') #by default flask finds templates in the template folder

@app.route('/Consent/', methods=['get', 'post'])
def consent():
    message =''
    if request.method == 'POST':        #if submit button clicked
        yes = request.form.get('agree') #get checkbox input (1/NULL)
        if yes=='1':                    #if box is checked
            message=''                  #clear the error message if it was there
            return redirect(f'/SurveyDemo/') #go to the survey page
        else:
            message = 'you must consent to continue'

    return render_template('Consent.html', message=message) 
@app.route('/Result/', methods=['get', 'post'])
def result():
    message='Sumitted Successfully: You are not at risk for heart disease.' #after algo integration, add another if else statement here for at risk/not at risk!
    if request.method == 'POST':
        return redirect(f'/') #go to homepage
    else: 
        return render_template('Result.html', message=message)

@app.route('/SurveyDemo/', methods=['get', 'post'])
def form():
    message =''
    if request.method == 'POST':
        sex = request.form.get('sex') #access data in form
        username = request.form.get('username') #use to personalize results page or prevent re-submission
        cp1 = request.form.get('cp1')
        cp2 = request.form.get('cp2')
        cp3 = request.form.get('cp3')
        
        if cp1=='on' and cp2=='on' and cp3=='on': #all boxes
            ChestPainType=4
        elif cp1!='on' and cp2!='on' and cp3!='on': #no boxes
            ChestPainType=1
        elif (cp1=='on' and cp2=='on') or (cp1=='on' and cp3=='on') or (cp2=='on' and cp3=='on'):  #two of three boxes checked
            ChestPainType=3
        else: 
            ChestPainType=2 #one box
        
        if cp2!='on':
            cp2=0
        else: cp2=1

        if sex =='M' or sex=='F' or sex=='X':  #check for valid sex input - can add checks to other inputs later.
            message = "submitted successfully"
            
            if sex == 'M':
                sex=1
            else: sex=0  #for now, deal with intersex/other by binning as F.  
            #write form data to a .csv file:
            with open('sampleform.csv', 'w', newline='') as csvfile:
                formdata = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                formdata.writerow(['age', 'sex', 'ChestPainType', 'RestingBP', 'MaxHR', 'ExerciseAngina'])
                formdata.writerow([request.form.get('age'), sex, ChestPainType, request.form.get('bp'), request.form.get('HR'), cp2])
            
            #import uploaded .csv files to local folder
            ECG_rest = request.files['ECG_rest']
            ECG_exercise = request.files['ECG_exercise']
            if ECG_rest.filename !='': #check if a file was uploaded
                ECG_rest.save(ECG_rest.filename) #save locally
            if ECG_exercise.filename !='':
                ECG_exercise.save(ECG_exercise.filename)

            return redirect ('/Result/')  #after form submission and input checking, send user to the Results page.
        else:
            message = "invalid input.  Please try again"
        
    return render_template('SurveyDemo.html', message=message)


#...

#start the local development server - if we get a webserver running comment this part out
if __name__ == "__main__":
    app.run(debug=True) #enable debugging - error messages will show in browser.
    #navigate to http://127.0.0.1:5000/ or http://localhost:5000/ in browser to see outputs