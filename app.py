from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    df = pd.read_csv("data/yama_pitching_data")
    unique_batters = df['batter_name'].unique()
    return render_template("index.html", batters=unique_batters)

if __name__ == '__main__':
    app.run(debug=True)