import os

from sklearn.naive_bayes import *
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv
app = Flask(__name__)
global Classifier
global Vectorizer
sms1 = " "

# load data
data = pandas.read_csv('spam.csv', encoding='latin-1')
test_data = data[:1172] # 4400 items
train_data = data[1172:] # 1172 items

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
# Classifier = BernoulliNB()
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)


@app.route('/result', methods=['GET'])
def index(text):
    message = request.args.get('message', text)
    error = ''
    predict_proba = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
          predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    # return jsonify(
    #           message=message, predict_proba=predict_proba,
    #           predict=predict, error=error)
    return predict

@app.route('/index')
def my_form():
    return render_template('fe.html')

@app.route('/home/', methods=['GET', 'POST'])
def home():
    sms = request.form['text']
    global sms1
    sms1 = sms
    predict = index(sms)
    return render_template('fin.html', predict=predict)

@app.route('/add/',methods=['GET', 'POST'])
def entry():

    if "ham" in request.form:
        str = "ham " + sms1
        with open(r'spam.csv', 'a', newline='') as csvfile:
            fieldnames = ['v1', 'v2', ',', ',', ',']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'v1': 'ham','v2': sms1})  

        return render_template('add.html')
 
    elif "spam" in request.form:
        with open(r'spam.csv', 'a', newline='') as csvfile:
            fieldnames = ['v1', 'v2', ',', ',', ',']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'v1': 'spam','v2': sms1})
            
        return render_template('add.html')
 
    elif "back" in request.form:
        return render_template('fe.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)