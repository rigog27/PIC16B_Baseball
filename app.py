from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    df = pd.read_csv("data/yama_pitching_data")
    unique_batters = df['batter_name'].unique()
    return render_template("index.html", batters=unique_batters)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    batter = request.form.get("batter")
    outs = request.form.get("outs")
    strikes = request.form.get("strikes")
    balls = request.form.get("balls")
    on_1b = request.form.get("on_1b")
    on_2b = request.form.get("on_2b")
    on_3b = request.form.get("on_3b")

    return render_template("prediction.html", outs=outs,
                           strikes=strikes, batter=batter, balls=balls,
                           on_1b=on_1b, on_2b=on_2b, on_3b=on_3b)


if __name__ == '__main__':
    app.run(debug=True)