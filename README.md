# Production-Based-ML-portfolio

I created a basic flask-based portal where a person can give input on 14 parameters. I integrated a classification model to this project which predicts on these input data and lets the person knows if there is a risk if heart disease or not. 


The dataset is taken from - https://www.kaggle.com/ronitf/heart-disease-uci


## Installation

Install with pip:

```
$ pip install -r requirements.txt
```

## Flask Application Structure 
```
.
.
├── app.py
├── dataset
│   └── HeartDiseaseDataset.csv
├── JupyterNotebook
│   ├── Heart Disease Dataset.csv
│   └── HeartDiseases.ipynb
├── requirements.txt
├── resource
│   └── diseaseprediction.joblib
├── templates
    └── home.html



```


## Flask Configuration

#### Example

```
app = Flask(__name__)
app.config['DEBUG'] = True
```
### Configuring From Files

#### Example Usage

```
app = Flask(__name__ )
app.config.from_pyfile('config.Development.cfg')
```

## Run Flask
### Run flask for develop
```
$ python app.py
```

#### cfg example

```

##Flask settings
DEBUG = True  # True/False
TESTING = False


