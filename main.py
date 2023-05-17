from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/inicio")
def home():
    return render_template('index.html')


@app.route("/information")
def information():
    return render_template('information.html')


app.run(debug=True)
