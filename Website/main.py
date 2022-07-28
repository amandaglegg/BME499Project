# Main webpage demo code.  Handle form data from website and call the algorithm.
# Author: Victoria Hartman, initially adapted from https://developer.mozilla.org/en-US/docs/Learn/Forms/Sending_and_retrieving_form_data 
# and https://overiq.com/flask-101/form-handling-in-flask/
# Last Updated July 2022

from asyncio.windows_events import NULL
from pickle import TRUE
from flask import Flask, render_template, request, redirect, url_for 
import Backend
import csv
import os 

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
            #write form data to a .csv file in Website folder:
            with open('C://Users/vh1_2/Documents/GitHub/BME499Project/Website/sampleform.csv', 'w', newline='') as csvfile:
                formdata = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                formdata.writerow(['age', 'sex', 'chest pain type', 'resting bp s', 'max heart rate', 'exercise angina'])
                formdata.writerow([request.form.get('age'), sex, ChestPainType, request.form.get('bp'), request.form.get('HR'), cp2])
            
            #import uploaded .csv files to local folder
            ECG_rest = request.files['ECG_rest']
            ECG_exercise = request.files['ECG_exercise']
            if ECG_rest.filename !='': #check if a file was uploaded
                ECG_rest.save(ECG_rest.filename) #save locally
            if ECG_exercise.filename !='':
                ECG_exercise.save(ECG_exercise.filename)
            os.rename(ECG_rest.filename, 'C://Users/vh1_2/Documents/GitHub/BME499Project/Website/pre_exercise_ecg.txt') #change the filename and move to Website folder
            os.rename(ECG_exercise.filename, 'C://Users/vh1_2/Documents/GitHub/BME499Project/Website/post_exercise_ecg.txt')
            return redirect ('/Result/')  #after form submission and input checking, send user to the Results page.
        else:
            message = "invalid input.  Please try again"
        
    return render_template('SurveyDemo.html', message=message)

@app.route('/Result/', methods=['get', 'post'])
def result():
    our_path = os.path.abspath(os.curdir)
    preexercise_path = our_path + '/Website/pre_exercise_ecg.txt'
    Backend.ecg_plot(preexercise_path)
    try: 
        if Backend.heartdisease() == 0: 
            message = 'Submitted Successfully. You are not at risk for heart disease.' #after algo integration, add another if else statement here for at risk/not at risk!
        elif Backend.heartdisease() == 1:
            message = 'Submitted Successfully. You are at risk for heart disease.'
    except:
        message = 'Unable to process your ECG reading, you may need to retake your ECG reading for a clearer result'
    if request.method == 'POST':
        os.remove('C://Users/vh1_2/Documents/GitHub/BME499Project/Website/post_exercise_ecg.txt') #delete the ecg files
        os.remove('C://Users/vh1_2/Documents/GitHub/BME499Project/Website/pre_exercise_ecg.txt')
        os.remove('C://Users/vh1_2/Documents/GitHub/BME499Project//Website/static/Peaks.jpeg')
        return redirect(f'/') #go to homepage
    else: 
        with open('C://Users/vh1_2/Documents/GitHub/BME499Project/Website/sampleform.csv', 'r') as sampleform:
            row=[] #intialize lists
            fields=[]
            survey=csv.reader(sampleform) #creates a csvreader object
            fields=next(survey) #extract field names from first row
            row=next(survey) #get row data
            if int(row[2]) > 1: #if there is any chest pain
                painmessage='You reported some chest pain. Contact your doctor for further investigation.'
            else: painmessage='Other risk factors for heart disease include smoking, drinking alcohol, diabetes, poor diet and lack of exercise.'
            if int(row[3]) < 90:
                bpmessage='You have low blood pressure and might be at risk for hypotension. You may feel lightheaded, weak, dizzy, or even faint. It can be caused by not getting enough fluids, blood loss, some medical conditions, or medications, including those prescribed for high blood pressure.'
            elif int(row[3]) <120:
                bpmessage='Your blood pressure is in the normal range.'
            elif int(row[3]) <129:
                bpmessage='Your blood pressure is elevated.  People with elevated blood pressure are likely to develop high blood pressure unless steps are taken to control the condition.'
            elif int(row[3])<139:
                bpmessage='You might be at risk for Hypertension Stage 1.  Doctors are likely to prescribe lifestyle changes and may consider adding blood pressure medication based on your risk of atherosclerotic cardiovascular disease (ASCVD), such as heart attack or stroke.'
            elif int(row[3])<180:
                bpmessage='You might be at risk for Hypertension Stage 2.  Doctors are likely to prescribe a combination of blood pressure medications and lifestyle changes.'
            else: bpmessage='You might be at risk for Hypertensive Crisis - wait five minutes and test your blood pressure again.  If it remains high, seek medical attention immediately. If your blood pressure is higher than 180/120 mm Hg and you are experiencing signs of possible organ damage such as chest pain, shortness of breath, back pain, numbness/weakness, change in vision or difficulty speaking, do not wait to see if your pressure comes down on its own. Call 911.'
            
            goodhr=220-int(row[0]) #calculate ideal hr from age
            if int(row[4]) <= goodhr:
                hrmessage='Your heart rate is in the healthy range.'
            else: hrmessage='Your heart rate is above the healthy limit: consult a doctor or consider lifestyle changes.'
        return render_template('Result.html', message=message, age=row[0], restingbp=row[3], hr=row[4], goodhr=goodhr, bpmessage=bpmessage, hrmessage=hrmessage, painmessage=painmessage) #render the results page and pass in survey data
#...

#start the local development server - if we get a webserver running comment this part out
if __name__ == "__main__":
    app.run(debug=True) #enable debugging - error messages will show in browser.
    #navigate to http://127.0.0.1:5000/ or http://localhost:5000/ in browser to see outputs
