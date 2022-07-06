#receive form data from website
#adapted from https://developer.mozilla.org/en-US/docs/Learn/Forms/Sending_and_retrieving_form_data 
#https://overiq.com/flask-101/form-handling-in-flask/

from asyncio.windows_events import NULL
from pickle import TRUE
from flask import Flask, render_template, request, redirect
import csv

from numpy import True_
app = Flask(__name__)

@app.route('/', methods=['get', 'post'])
def index():
    if request.method == 'POST':
        return redirect(f'/Consent/') #go to the consent page   
    
    return render_template('index.html', name='homepage') #by default flask finds templates in the template folder

@app.route('/Consent/', methods=['get', 'post'])
def consent():
    message =''
    if request.method == 'POST':
        yes = request.form.get('agree')
        if yes=='1': #if box is checked
            message=''
            return redirect(f'/SurveyDemo/') #go to the survey page
        else:
            message = 'you must consent to continue'

    return render_template('Consent.html', message=message) 
@app.route('/Result/', methods=['get'])
def result():
    message=''
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

        #note: use request.files.get to access uploaded files.
        
        if sex =='M' or sex=='F' or sex=='X':
            message = "submitted successfully:  You are not at risk for heart disease"
            #after algo integration, add another if else statement here for at risk/not at risk!

            #write form data to a .csv file:
            with open('sampleform.csv', 'w', newline='') as csvfile:
                formdata = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                formdata.writerow(['age', 'sex', 'ChestPainType', 'RestingBP', 'MaxHR', 'ExerciseAngina'])
                formdata.writerow([request.form.get('age'), request.form.get('sex'), ChestPainType, request.form.get('bp'), request.form.get('HR'), cp2])

            ECG_rest = request.files['ECG_rest']
            ECG_exercise = request.files['ECG_exercise']
            if ECG_rest.filename !='':
                ECG_rest.save(ECG_rest.filename)
            if ECG_exercise.filename !='':
                ECG_exercise.save(ECG_exercise.filename)

            return render_template('Result.html', message=message)
        else:
            message = "invalid input.  Please try again"
        
    return render_template('SurveyDemo.html', message=message)


#...

#start the development server
if __name__ == "__main__":
    app.run(debug=True)
    #navigate to http://127.0.0.1:5000/ or http://localhost:5000/ in browser to see outputs