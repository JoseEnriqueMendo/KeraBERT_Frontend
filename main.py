from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/inicio")
def home():
    return render_template('index.html')


@app.route("/information")
def information():
    return render_template('information.html')


@app.route("/pentacam")
def pentacam():
    return render_template('pentacam.html')


@app.route("/casoClinico")
def casoClinico():
    return render_template('casoClinico.html')


app.run(debug=True)
