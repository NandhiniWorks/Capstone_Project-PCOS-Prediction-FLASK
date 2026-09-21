from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("pcos_logistic_model.pkl")
scaler = joblib.load("pcos_scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        cycle = request.form["cycle"]
        beta_hcg_ii = float(request.form["beta_hcg_ii"])
        hair_growth = request.form["hair_growth"]
        follicle_left = float(request.form["follicle_left"])
        follicle_right = float(request.form["follicle_right"])

        cycle_value = 1 if cycle == "Irregular" else 0
        hair_value = 1 if hair_growth == "Yes" else 0

        new_patient = pd.DataFrame({
            "Cycle(R/I)": [cycle_value],
            "Beta-HCG-II": [beta_hcg_ii],
            "Hair Growth": [hair_value],
            "Follicle Left": [follicle_left],
            "Follicle Right": [follicle_right]
        })

        new_patient_scaled = scaler.transform(new_patient)

        result = model.predict(new_patient_scaled)

        if result[0] == 1:
            prediction = "PCOS predicted"
        else:
            prediction = "No PCOS predicted"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)