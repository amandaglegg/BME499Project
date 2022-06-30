#receive form data from website
#adapted from https://developer.mozilla.org/en-US/docs/Learn/Forms/Sending_and_retrieving_form_data 
#https://overiq.com/flask-101/form-handling-in-flask/

from flask import Flask, render_template, request, redirect
import csv
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

@app.route('/SurveyDemo/', methods=['get', 'post'])
def form():
    message =''
    if request.method == 'POST':
        sex = request.form.get('sex') #access data in form
        #note: use request.files.get to access uploaded files.
        
        if sex =='M' or sex=='F' or sex=='X':
            message = "submitted successfully:  You are not at risk for heart disease"
            #after algo integration, add another if else statement here for at risk/not at risk!

            #write form data to a .csv file:
            with open('sampleform.csv', 'w', newline='') as csvfile:
                formdata = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                formdata.writerow(['username','age', 'sex'])
                formdata.writerow([request.form.get('username'), request.form.get('age'), request.form.get('sex')])

            return render_template('Result.html', message=message)
        else:
            message = "invalid input.  Please try again"
        
    return render_template('SurveyDemo.html', message=message)
#...

#start the development server
if __name__ == "__main__":
    app.run(debug=True)
    #navigate to http://127.0.0.1:5000/ or http://localhost:5000/ in browser to see outputs