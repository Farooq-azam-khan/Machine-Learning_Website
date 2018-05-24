from flask import Flask, render_template, url_for, redirect, request, flash
from iris_forms import DataPredict
import numpy as np
# graph
import pygal
# stypes
from pygal.style import DarkSolarizedStyle
from pygal.style import NeonStyle
# data
from sklearn import datasets

# model evaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics

# models
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess' # temp key

@app.route("/")
def home():
    return render_template("home.html", title="home")

@app.route("/iris", methods=['GET', 'POST'])
def iris():
    # load data
    raw_data = datasets.load_iris()
    features = raw_data.data
    labels = raw_data.target
    # splitting
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.4, random_state=4)

    # dict of vals
    models = {'knn':KNeighborsClassifier,
            'logistic_regression':LogisticRegression,
            'NuSVC': svm.NuSVC,
            'SVC': svm.SVC,
            'Ada_boostClassifier': ensemble.AdaBoostClassifier
            }
    scores = {}

    for key, val in models.items():
        model = val()
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        scores[key] = accuracy

    # graph data of score
    title = "Iris Data set graphed"
    bar_chart = pygal.Bar(width=1200, height=600, explicit_size=True, title=title, style=NeonStyle)
    ml_models_used = [key for key, val in models.items()]
    bar_chart.x_labels = ml_models_used
    bar_chart.add('accuracy', [val for key,val in scores.items()])

    # graph data of vals
    measurements_graph = pygal.XY(width=800, height=400, stroke=False)
    measurements_graph.title = "Iris Data"
    measurements_graph.add('pl vs pw', [(val[0], val[1]) for val in features])
    measurements_graph.add('pl vs sw', [(val[2], val[3]) for val in features])
    measurements_graph.add('pl vs  sl', [(val[0], val[2]) for val in features])
    measurements_graph.add('pw vs sw', [(val[1], val[3]) for val in features])

    # get vals for prediction
    form = DataPredict()
    if form.validate_on_submit():
        pred_y = np.array([[form.pedal_length.data,form.pedal_width.data,form.sepal_length.data,form.sepal_width.data]])
        msg = ""
        for key, val in models.items():
            model = val()
            model.fit(train_X, train_y)
            pred_val = model.predict(pred_y)
            msg += "alg: {} acc: {}, ".format(key, pred_val[0])
        flash(msg, "success")
        return redirect(url_for('iris'))

    return render_template("iris.html",features=features, labels=labels, scores=scores, barchart=bar_chart, measurements_graph=measurements_graph, form=form)

@app.route("/iris/data")
def iris_data():
    # load data
    raw_data = datasets.load_iris()
    features = raw_data.data
    labels = raw_data.target
    length = len(features)
    return render_template("iris_data.html", features=features, labels=labels, length=length)

@app.route("/titanic")
def titanic():
    return render_template("titanic.html")

@app.route("/titanic/data")
def titanic_data():
    return render_template("titanic_data.html")

@app.route("/neural_network")
def neural_network():
    return render_template("neural_network.html", title="neural_network")

@app.route("/resources")
def resources():
    return render_template("resources.html")

if __name__ == '__main__':
    app.run(debug=True)
