import pandas as pd
from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)
pipeline = load("resource/diseaseprediction.joblib")


@app.route('/', methods=['GET'])
def home():  # put application's code here
    return render_template('home.html')


# checking form ImmutableMultiDict
# ([('age', '35'), ('female', '0'), ('CP', '2'), ('trestbps', '115'), ('trestbps', '245'), ('RECG', '0'), ('thalach', '147'), ('no', '0'), ('oldpeak', '0.4'), ('Slope', '2'), ('CA', '0'), ('THAL', '2')])

# checking form {'age': '56', 'sex': '1', 'cp': '2', 'trestbps': '130', 'chol': '256', 'fbs': '1', 'restecg': '0', 'thalach': '142', 'exang': '1', 'oldpeak': '0.6', 'slope': '1', 'ca': '1', 'thal': '1'}

@app.route('/', methods=['GET', 'POST'])
def inputForm():
    print("type", type(request.form.to_dict(flat=False)))
    print("checking form", request.form.to_dict())
    data = request.form.to_dict()
    df = pd.DataFrame(data, index=[0])
    print("prediction", pipeline.predict(df))

    if pipeline.predict(df) == [1]:
        prediction = "You are in risk of heart disease"

    elif pipeline.predict(df) == [0]:
        prediction = "You don't have risk of heart disease"

    else:
        prediction = "Can't predict anything"


    return render_template('home.html', prediction = prediction, show_predictions_modal = True )

if __name__ == "__main__":
    app.run()
