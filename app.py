import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, send_file
from io import BytesIO
import os

app = Flask(__name__)

# Load trained model and scaler with error handling
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

# Load dataset with error handling
try:
    df = pd.read_csv("Cost_of_Living_Index_2022.csv")
except FileNotFoundError:
    print("Dataset file not found.")
    df = None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded properly"})

    try:
        # Get input values safely
        rent = float(request.form.get("rent", 0))
        living_plus_rent = float(request.form.get("living_plus_rent", 0))
        groceries = float(request.form.get("groceries", 0))
        restaurant = float(request.form.get("restaurant", 0))
        purchasing_power = float(request.form.get("purchasing_power", 0))

        # Validate input
        if any(value < 0 for value in [rent, living_plus_rent, groceries, restaurant, purchasing_power]):
            return jsonify({"error": "Values must be non-negative"})

        # Prepare and scale input data
        input_data = np.array([[rent, living_plus_rent, groceries, restaurant, purchasing_power]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]

        return jsonify({"result": round(prediction, 2)})

    except ValueError:
        return jsonify({"error": "Invalid input values. Please enter valid numbers."})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/plot/<plot_type>")
def generate_plot(plot_type):
    if df is None:
        return jsonify({"error": "Dataset not found"})

    plt.figure(figsize=(8, 5))

    try:
        required_columns = {
            "cost_vs_rent": ["Rent Index", "Cost of Living Index"],
            "cost_vs_groceries": ["Groceries Index", "Cost of Living Index"],
            "top_10_expensive": ["Country", "Cost of Living Index"],
            "top_10_cheap": ["Country", "Cost of Living Index"]
        }

        # Check if necessary columns exist
        if plot_type in required_columns and not all(col in df.columns for col in required_columns[plot_type]):
            return jsonify({"error": "Missing necessary columns in dataset"})

        if plot_type == "cost_vs_rent":
            plt.scatter(df["Rent Index"], df["Cost of Living Index"], color="blue", alpha=0.5)
            plt.xlabel("Rent Index")
            plt.ylabel("Cost of Living Index")
            plt.title("Cost of Living vs Rent Index")

        elif plot_type == "cost_vs_groceries":
            plt.scatter(df["Groceries Index"], df["Cost of Living Index"], color="green", alpha=0.5)
            plt.xlabel("Groceries Index")
            plt.ylabel("Cost of Living Index")
            plt.title("Cost of Living vs Groceries Index")

        elif plot_type == "top_10_expensive":
            top_10_expensive = df.nlargest(10, "Cost of Living Index")
            plt.barh(top_10_expensive["Country"], top_10_expensive["Cost of Living Index"], color="red")
            plt.xlabel("Cost of Living Index")
            plt.ylabel("Country")
            plt.title("Top 10 Most Expensive Countries")
            plt.gca().invert_yaxis()

        elif plot_type == "top_10_cheap":
            top_10_cheap = df.nsmallest(10, "Cost of Living Index")
            plt.barh(top_10_cheap["Country"], top_10_cheap["Cost of Living Index"], color="green")
            plt.xlabel("Cost of Living Index")
            plt.ylabel("Country")
            plt.title("Top 10 Least Expensive Countries")
            plt.gca().invert_yaxis()

        else:
            return jsonify({"error": "Invalid plot type"}), 400

        # Save plot to BytesIO and return as response
        img = BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        plt.close()  # Prevent memory leak
        img.seek(0)

        return send_file(img, mimetype="image/png")

    except KeyError:
        return jsonify({"error": "Missing necessary columns in dataset"})


if __name__ == "__main__":
    app.run(debug=True)
