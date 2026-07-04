from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Trained Model
model = joblib.load("customer_churn_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        tenure = float(request.form["tenure"])
        monthly = float(request.form["monthly"])
        total = float(request.form["total"])

        data = np.array([[tenure, monthly, total]])

        prediction = model.predict(data)

        if prediction[0] == 1:
             result = "Customer is likely to CHURN ❌"
        else:
            result = "Customer is likely to STAY ✅"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True) 